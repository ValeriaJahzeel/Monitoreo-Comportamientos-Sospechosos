"""
Tracker multi-objeto estilo SORT (Simple Online and Realtime Tracking),
implementado desde cero con numpy + scipy (sin dependencias externas nuevas).

Reemplaza al tracker original de objectDetection.py, que solo asociaba
detecciones por distancia euclidiana entre centroides del frame anterior al
actual. Ese enfoque no tolera oclusiones breves ni saltos de posición: en
cuanto una persona se pierde un frame (por ejemplo porque YOLO no la detectó
con suficiente confianza, algo común en video de vigilancia con personas
lejos de cámara), se le asigna un ID nuevo en vez de recuperar el anterior.
Esto generaba decenas de IDs "fantasma" por clip de pocos segundos y
trayectorias demasiado cortas para que analizar_trayectoria/detectar_
comportamiento_ciclico pudieran calcular nada útil (necesitan varios puntos
de historia por objeto).

Este tracker resuelve eso con dos ideas del SORT original (Bewley et al.,
2016):
  1. Un filtro de Kalman de velocidad constante por track, que predice dónde
     debería estar cada objeto en el frame actual aunque no haya detección
     nueva que lo confirme (tolera oclusiones cortas hasta `max_age` frames).
  2. Asociación óptima detección<->track por IoU (Hungarian algorithm), en
     vez de la comparación greedy por distancia de centroide, que puede
     quedar atrapada en asignaciones subóptimas cuando hay varias personas
     cerca entre sí.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def bbox_a_estado(bbox):
    """Convierte [x1, y1, x2, y2] a [u, v, s, r] (centro, área, aspecto)."""
    x1, y1, x2, y2 = bbox
    w = max(x2 - x1, 1e-3)
    h = max(y2 - y1, 1e-3)
    u = x1 + w / 2.0
    v = y1 + h / 2.0
    s = w * h
    r = w / h
    return np.array([u, v, s, r])


def estado_a_bbox(estado):
    """Convierte [u, v, s, r] de vuelta a [x1, y1, x2, y2]."""
    u, v, s, r = estado
    s = max(s, 1e-3)
    r = max(r, 1e-3)
    w = np.sqrt(s * r)
    h = s / w
    return np.array([u - w / 2.0, v - h / 2.0, u + w / 2.0, v + h / 2.0])


def iou(bbox1, bbox2):
    """Intersection-over-Union entre dos bounding boxes [x1, y1, x2, y2]."""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    interseccion = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, bbox1[2] - bbox1[0]) * max(0.0, bbox1[3] - bbox1[1])
    area2 = max(0.0, bbox2[2] - bbox2[0]) * max(0.0, bbox2[3] - bbox2[1])
    union = area1 + area2 - interseccion

    return interseccion / union if union > 0 else 0.0


class KalmanBoxTracker:
    """
    Un track individual: estado [u, v, s, r, u', v', s'] con modelo de
    velocidad constante (r se asume sin velocidad propia, como en SORT).
    """

    _siguiente_id = 0

    def __init__(self, bbox):
        # Transición de estado: posición += velocidad
        self.F = np.eye(7)
        self.F[0, 4] = 1
        self.F[1, 5] = 1
        self.F[2, 6] = 1

        # Matriz de observación: solo medimos [u, v, s, r]
        self.H = np.zeros((4, 7))
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = self.H[3, 3] = 1

        # Ruido de proceso y de medición (valores estándar usados en SORT)
        self.Q = np.eye(7)
        self.Q[4:, 4:] *= 0.01
        self.Q[2, 2] *= 0.01  # incertidumbre baja en la velocidad de área

        self.R = np.eye(4)
        self.R[2:, 2:] *= 10.0  # más ruido tolerado en área/aspecto

        self.P = np.eye(7)
        self.P[4:, 4:] *= 1000.0  # alta incertidumbre inicial en velocidad
        self.P *= 10.0

        self.x = np.zeros(7)
        self.x[:4] = bbox_a_estado(bbox)

        self.id = KalmanBoxTracker._siguiente_id
        KalmanBoxTracker._siguiente_id += 1

        self.time_since_update = 0
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def predict(self):
        if self.x[6] + self.x[2] <= 0:  # el área no puede ser negativa
            self.x[6] = 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        return estado_a_bbox(self.x[:4])

    def update(self, bbox):
        z = bbox_a_estado(bbox)

        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P

        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

    def bbox_actual(self):
        return estado_a_bbox(self.x[:4])


class Sort:
    """
    Tracker multi-objeto. Uso:
        tracker = Sort(max_age=15, min_hits=1, iou_threshold=0.3)
        ids_bboxes = tracker.update(detecciones)  # Nx4 [x1,y1,x2,y2]

    max_age: frames que un track sobrevive sin ser confirmado por una
        detección nueva antes de eliminarse (tolerancia a oclusión).
    min_hits: detecciones consecutivas necesarias antes de reportar un track
        como confirmado. Con min_hits=1 se reporta desde la primera vez que
        se ve, igual que el tracker anterior (más simple de integrar en el
        resto del pipeline, que espera un bbox por objeto detectado).
    iou_threshold: IoU mínimo para considerar que una detección corresponde
        a un track existente.
    """

    def __init__(self, max_age=15, min_hits=1, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = []

    def _asociar(self, detecciones, predicciones):
        if len(predicciones) == 0 or len(detecciones) == 0:
            return [], list(range(len(detecciones))), list(range(len(predicciones)))

        matriz_iou = np.zeros((len(detecciones), len(predicciones)))
        for d, det in enumerate(detecciones):
            for t, pred in enumerate(predicciones):
                matriz_iou[d, t] = iou(det, pred)

        filas, columnas = linear_sum_assignment(-matriz_iou)

        emparejados = []
        detecciones_sin_asignar = list(range(len(detecciones)))
        tracks_sin_asignar = list(range(len(predicciones)))

        for d, t in zip(filas, columnas):
            if matriz_iou[d, t] < self.iou_threshold:
                continue
            emparejados.append((d, t))
            detecciones_sin_asignar.remove(d)
            tracks_sin_asignar.remove(t)

        return emparejados, detecciones_sin_asignar, tracks_sin_asignar

    def update(self, detecciones):
        """
        detecciones: array Nx4 de [x1, y1, x2, y2].
        Devuelve {track_id: (x1, y1, x2, y2)} solo para tracks confirmados
        y vistos en este frame.
        """
        predicciones = [t.predict() for t in self.tracks]

        emparejados, det_sin_asignar, tracks_sin_asignar = self._asociar(
            detecciones, predicciones
        )

        for d, t in emparejados:
            self.tracks[t].update(detecciones[d])

        for d in det_sin_asignar:
            self.tracks.append(KalmanBoxTracker(detecciones[d]))

        # Eliminar tracks que llevan demasiados frames sin confirmarse
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        resultado = {}
        for t in self.tracks:
            if t.time_since_update == 0 and t.hit_streak >= self.min_hits:
                x1, y1, x2, y2 = t.bbox_actual()
                resultado[f"obj_{t.id}"] = (int(x1), int(y1), int(x2), int(y2))

        return resultado
