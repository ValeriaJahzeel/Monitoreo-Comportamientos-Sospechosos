"""
Módulo de detección de objetos para análisis de movimientos sospechosos.
Versión optimizada que aprovecha FeatureExtractor.
"""
import cv2
import numpy as np
import os
import torch
from ultralytics import YOLO
import featureExtraction as fe
from tracking import Sort

class ObjectDetector:
    def __init__(self, model_path=None):
        # Por defecto, usar los pesos que viven junto a este script
        # (evita rutas absolutas atadas a la máquina de un desarrollador)
        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 'yolo-Weights', 'yolov8n.pt'
            )
        # Inicializar modelo YOLO. Se fija el device explícitamente (en vez
        # de dejar que ultralytics lo infiera) para garantizar que use GPU
        # cuando esté disponible.
        self.modelo = YOLO(model_path)
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        # Inicializar extractor de características
        self.feature_extractor = fe.FeatureExtractor(history_size=30)
        # Tracker multi-objeto (Kalman + asociación por IoU). max_age=15
        # tolera hasta ~0.5s de oclusión/detección perdida a 30fps antes de
        # abandonar un track; min_hits=1 reporta un objeto desde la primera
        # vez que se ve, igual que el tracker anterior.
        self.tracker = Sort(max_age=15, min_hits=1, iou_threshold=0.3)
        # Variables para seguimiento
        self.frame_anterior = None
        self.frame_num = 0
        self.objetos_previos = {}

    def detectar_objetos(self, frame):
        """Detecta objetos usando YOLO y retorna los bounding boxes con IDs consistentes"""
        # classes=[0] restringe la detección a "person" (COCO). Sin este
        # filtro YOLO detecta y trackea las 80 clases de COCO (sillas,
        # autos, botellas...) como si fueran personas, generando decenas de
        # IDs espurios por video. Se deja el confidence threshold en el
        # default de ultralytics (0.25): subirlo (probado en 0.4) reducía
        # aún más el ruido pero perdía casi todas las detecciones en clips
        # con personas lejos de cámara (típico en merodeo).
        resultados = self.modelo(frame, stream=True, classes=[0], verbose=False, device=self.device)
        detecciones = []

        for r in resultados:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detecciones.append((x1, y1, x2, y2))

        detecciones_array = np.array(detecciones, dtype=float) if detecciones else np.empty((0, 4))

        # El tracker asocia estas detecciones con los tracks existentes
        # (Kalman + IoU) y devuelve IDs consistentes entre frames, incluso
        # si el objeto no se detectó en uno o varios frames intermedios.
        bboxes = self.tracker.update(detecciones_array)

        # Se preserva self.objetos_previos porque limpiar_objetos_perdidos()
        # y predecir_posicion() lo usan para saber qué IDs siguen activos
        self.objetos_previos = {}
        for obj_id, bbox in bboxes.items():
            x1, y1, x2, y2 = bbox
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            self.objetos_previos[obj_id] = (x1, y1, x2, y2, cx, cy)

        return bboxes
    
    def determinar_roi_flujo(self, flujo, umbral=1.0, densidad_min=10, filtro_tamano=5, min_area=500):
        """
        Determina una región de interés basada en el flujo óptico con reducción de falsos positivos
        """
        # Calcular magnitud del flujo
        magnitud, _ = cv2.cartToPolar(flujo[..., 0], flujo[..., 1])
        
        # Crear máscara binaria de puntos con flujo significativo
        mascara = np.zeros_like(magnitud, dtype=np.uint8)
        mascara[magnitud > umbral] = 255
        
        # Filtros morfológicos para eliminar ruido
        kernel = np.ones((filtro_tamano, filtro_tamano), np.uint8)
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)  # Elimina ruido pequeño
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)  # Cierra huecos
        
        # Encontrar contornos en la máscara
        contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Si no hay contornos significativos, devolver None
        if not contornos:
            return None
        
        # Filtrar contornos por área y obtener el mayor
        contornos_validos = [cnt for cnt in contornos if cv2.contourArea(cnt) > min_area]
        
        if not contornos_validos:
            return None
        
        # Encontrar contorno con mayor área (movimiento principal)
        mayor_contorno = max(contornos_validos, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(mayor_contorno)
        
        # Expandir ligeramente el ROI para asegurar que cubra todo el objeto
        expansion = 10
        x = max(0, x - expansion)
        y = max(0, y - expansion)
        w = min(flujo.shape[1] - x, w + 2*expansion)
        h = min(flujo.shape[0] - y, h + 2*expansion)
        
        return x, y, x + w, y + h
    
    def calcular_flujo_optico_denso(self, frame_prev, frame_curr):
        """Calcula el flujo óptico denso entre dos frames"""
        return cv2.calcOpticalFlowFarneback(
            frame_prev, frame_curr, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
    
    def visualizar_flujo_denso(self, img, flujo, paso=16):
        """Visualiza el flujo óptico denso sobre la imagen"""
        vis_img = img.copy()
        h, w = img.shape[:2]
        y, x = np.mgrid[0:h:paso, 0:w:paso].reshape(2, -1).astype(int)
        fx, fy = flujo[y, x].T
        
        # Crear líneas para representar el flujo
        lineas = np.vstack([x, y, x + fx * 3, y + fy * 3]).T.reshape(-1, 2, 2)
        lineas = np.int32(lineas + 0.5)
        
        # Dibujar las líneas de flujo
        for (x1, y1), (x2, y2) in lineas:
            cv2.arrowedLine(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 1, tipLength=0.3)
            
        return vis_img
    
    def limpiar_objetos_perdidos(self, max_frames_ausente=30):
        """Elimina objetos que no han sido detectados recientemente"""
        # Registrar el último frame en que se vio cada objeto
        frame_actual = self.frame_num
        
        if not hasattr(self, 'ultimo_frame_visto'):
            self.ultimo_frame_visto = {}
        
        # Actualizar último frame visto para objetos actuales
        for obj_id in self.objetos_previos:
            self.ultimo_frame_visto[obj_id] = frame_actual
        
        # Eliminar objetos antiguos de las estructuras de datos
        objetos_a_eliminar = []
        for obj_id, ultimo_frame in self.ultimo_frame_visto.items():
            if frame_actual - ultimo_frame > max_frames_ausente:
                objetos_a_eliminar.append(obj_id)
        
        # Eliminar de todas las estructuras de datos
        for obj_id in objetos_a_eliminar:
            if obj_id in self.ultimo_frame_visto:
                del self.ultimo_frame_visto[obj_id]
            if obj_id in self.feature_extractor.trayectorias:
                del self.feature_extractor.trayectorias[obj_id]
            if obj_id in self.feature_extractor.contador_permanencia:
                del self.feature_extractor.contador_permanencia[obj_id]
            
    def analizar_velocidad_sospechosa(self, historial_velocidades, umbral_varianza=10.0):
        """
        Analiza si un patrón de velocidad puede ser sospechoso
        basado en cambios bruscos o patrones inusuales
        """
        if len(historial_velocidades) < 10:
            return False, 0
        
        # Calcular estadísticas
        velocidad_media = np.mean(historial_velocidades)
        velocidad_std = np.std(historial_velocidades)
        coef_variacion = velocidad_std / (velocidad_media + 1e-6)
        
        # Detectar cambios bruscos de velocidad
        cambios = np.diff(historial_velocidades)
        cambios_grandes = np.abs(cambios) > (velocidad_media * 0.5)
        frecuencia_cambios = np.sum(cambios_grandes) / len(cambios)
        
        # Puntuación de sospecha
        puntuacion = coef_variacion * 5 + frecuencia_cambios * 10
        
        return puntuacion > umbral_varianza, puntuacion

    def predecir_posicion(self, obj_id, frames_adelante=1):
        """Predice la posición futura de un objeto basado en su trayectoria reciente"""
        if obj_id not in self.feature_extractor.trayectorias:
            return None
            
        trayectoria = list(self.feature_extractor.trayectorias[obj_id])
        if len(trayectoria) < 3:
            return trayectoria[-1] if trayectoria else None
        
        # Calcular vector de movimiento promedio de los últimos N frames
        n_frames = min(5, len(trayectoria) - 1)
        vectores = []
        
        for i in range(1, n_frames + 1):
            vector = np.array(trayectoria[-i]) - np.array(trayectoria[-(i+1)])
            vectores.append(vector)
        
        vector_promedio = np.mean(vectores, axis=0)
        
        # Predecir posición futura
        ultima_pos = np.array(trayectoria[-1])
        pos_predicha = ultima_pos + (vector_promedio * frames_adelante)
        
        return tuple(pos_predicha.astype(int)) 

    def validar_movimiento_persona(self, roi, flujo_data):
        """
        Valida si un ROI de movimiento probablemente corresponde a una persona
        basado en proporción, tamaño y patrón de movimiento
        """
        if roi is None:
            return False
        
        x_min, y_min, x_max, y_max = roi
        ancho = x_max - x_min
        alto = y_max - y_min
        
        # 1. Verificar proporción (las personas suelen ser más altas que anchas)
        proporcion = alto / ancho if ancho > 0 else 0
        if proporcion < 1.2 or proporcion > 4.0:  # Fuera de proporciones típicas humanas
            return False
        
        # 2. Verificar tamaño mínimo (para evitar detecciones pequeñas)
        area = ancho * alto
        if area < 3000:  # Ajustar según el tamaño esperado de personas en el video
            return False
        
        # 3. Analizar el patrón de movimiento dentro del ROI
        # Asegurar que el ROI está dentro de los límites del flujo
        y_min = max(0, y_min)
        y_max = min(flujo_data.shape[0], y_max)
        x_min = max(0, x_min)
        x_max = min(flujo_data.shape[1], x_max)
        
        # Verificar que el ROI tiene dimensiones válidas
        if y_min >= y_max or x_min >= x_max:
            return False
        
        roi_flujo = flujo_data[y_min:y_max, x_min:x_max]
        
        # Verificar que hay datos de flujo
        if roi_flujo.size == 0:
            return False
        
        magnitud, angulo = cv2.cartToPolar(roi_flujo[..., 0], roi_flujo[..., 1])
        
        # Filtrar regiones con flujo significativo
        magnitud_significativa = magnitud > 1.0
        if not np.any(magnitud_significativa):
            return False
        
        # Calcular histograma de dirección del flujo
        bins = 8
        hist = np.zeros(bins)
        
        for i in range(bins):
            lower = i * (2*np.pi/bins)
            upper = (i+1) * (2*np.pi/bins)
            hist[i] = np.sum((angulo >= lower) & (angulo < upper) & magnitud_significativa)
        
        # Normalizar histograma
        total = np.sum(hist)
        if total > 0:
            hist = hist / total
        else:
            return False
        
        # Calcular entropía (movimientos humanos tienen cierta estructura)
        # Baja entropía = movimiento estructurado, alta entropía = movimiento aleatorio
        entropia = -np.sum(hist * np.log2(hist + 1e-10))
        
        # Si la entropía es muy alta, probablemente sea ruido o movimiento no humano
        if entropia > 2.5:
            return False
        
        return True
    
    def procesar_frame(self, frame, dibujar=True):
        """Procesa un frame individual y devuelve resultados de análisis.

        `dibujar` controla si se generan las anotaciones visuales (cajas,
        IDs, trayectorias, flechas de flujo óptico). En procesamiento por
        lotes (sin mostrar/guardar video) se debe dejar en False: evita
        dibujar sobre cada frame y, sobre todo, evita calcular el flujo
        óptico denso (Farneback sobre el frame completo, la operación más
        costosa de todo el pipeline) salvo en el único caso en que
        realmente se usa: como respaldo cuando YOLO no detectó a nadie.
        """
        # Convertir a escala de grises para flujo óptico
        frame_gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.feature_extractor.update_frame_dimensions(frame.shape[:2])

        frame_anotado = frame.copy() if dibujar else None

        # Detectar objetos con YOLO primero
        bboxes = self.detectar_objetos(frame)

        # El flujo óptico denso solo se necesita si YOLO no encontró a
        # nadie (respaldo) o si hay que dibujarlo para visualización
        flujo_data = None
        if (not bboxes or dibujar) and self.frame_anterior is not None:
            flujo_data = self.calcular_flujo_optico_denso(self.frame_anterior, frame_gris)

        # Si no se detectaron objetos pero hay flujo, determinar ROI
        if not bboxes and flujo_data is not None:
            roi = self.determinar_roi_flujo(flujo_data, 
                                        umbral=1.5,          # Aumentar umbral 
                                        densidad_min=20,     # Más puntos de movimiento requeridos
                                        filtro_tamano=7,     # Filtrado más agresivo
                                        min_area=700)        # Área mínima mayor
            
            # Solo crear bounding box si se encontró un ROI válido
            if roi is not None:
                x_min, y_min, x_max, y_max = roi
                
                # Validar si el movimiento corresponde a una persona (si implementaste esta función)
                es_valido = True
                if hasattr(self, 'validar_movimiento_persona'):
                    es_valido = self.validar_movimiento_persona(roi, flujo_data)
                
                if es_valido:
                    # Crear un bounding box "virtual" para la región con movimiento
                    bboxes["movimiento_0"] = (x_min, y_min, x_max, y_max)
                    if dibujar:
                        cv2.rectangle(frame_anotado, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                        cv2.putText(frame_anotado, "Movimiento detectado", (x_min, y_min-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Recolectar todas las características disponibles
        datos_caracteristicas = self.feature_extractor.recolectar_caracteristicas(
            self.frame_num, bboxes, fps=25
        )

        if dibujar:
            # Visualizar objetos detectados y sus trayectorias
            for obj_id, bbox in bboxes.items():
                x1, y1, x2, y2 = bbox
                # Dibujar bounding box
                cv2.rectangle(frame_anotado, (x1, y1), (x2, y2), (255, 0, 0), 2)

                # Dibujar ID y datos
                centroide = self.feature_extractor.calcular_centroide(bbox)
                cv2.circle(frame_anotado, centroide, 4, (0, 0, 255), -1)
                cv2.putText(frame_anotado, f"ID: {obj_id}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # Dibujar trayectoria si existe
                if obj_id in self.feature_extractor.trayectorias:
                    trayectoria = list(self.feature_extractor.trayectorias[obj_id])
                    if len(trayectoria) > 1:
                        # Convertir puntos de la trayectoria
                        puntos = np.array(trayectoria, dtype=np.int32)
                        # Dibujar línea de trayectoria
                        for i in range(1, len(puntos)):
                            cv2.line(frame_anotado, tuple(puntos[i-1]),
                                tuple(puntos[i]), (0, 255, 255), 2)

            # Visualizar flujo óptico
            if flujo_data is not None and self.frame_anterior is not None:
                frame_anotado = self.visualizar_flujo_denso(frame_anotado, flujo_data)

        # Purgar objetos que llevan tiempo sin detectarse (evita que
        # objetos_previos y feature_extractor.trayectorias crezcan sin
        # límite en videos largos)
        self.limpiar_objetos_perdidos()

        # Actualizar para el próximo frame
        self.frame_anterior = frame_gris
        self.frame_num += 1
        
        return frame_anotado, datos_caracteristicas
    
    def guardar_caracteristicas(self, datos, csv_path):
        """Guarda las características extraídas en un archivo CSV"""
        # Verificar si existe el directorio
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        
        # Usar la función del extractor para guardar
        self.feature_extractor.guardar_datos_csv(csv_path, datos)
    
    def procesar_video(self, video_path, csv_path, mostrar=True, usar_threads=True, guardar_video=None):
        """Procesa un video completo con soporte para multi-threading.

        `guardar_video`: ruta opcional donde escribir el video anotado
        (cajas, IDs, trayectorias). Si se especifica, fuerza dibujar=True
        aunque `mostrar` sea False.
        """
        import threading
        from queue import Queue, Empty

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: No se pudo abrir el video {video_path}")
            return

        # Resetear variables. self.frame_num solo lo incrementa procesar_frame,
        # que corre en el hilo de trabajo cuando usar_threads=True: no debe
        # tocarse también desde el hilo principal (evita condición de carrera
        # y doble conteo de frames).
        self.frame_num = 0
        self.frame_anterior = None
        # Tracker nuevo por video: si se reutiliza el mismo ObjectDetector
        # para varios videos (como hace el bloque __main__), los tracks del
        # video anterior no deben poder "engancharse" a detecciones del
        # video siguiente.
        self.tracker = Sort(max_age=self.tracker.max_age, min_hits=self.tracker.min_hits,
                             iou_threshold=self.tracker.iou_threshold)

        video_writer = None
        if guardar_video:
            mostrar = True  # sin dibujar=True no hay nada que escribir
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            os.makedirs(os.path.dirname(os.path.abspath(guardar_video)), exist_ok=True)
            video_writer = cv2.VideoWriter(
                guardar_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (ancho, alto)
            )

        def guardar_resultado(datos, frame_anotado=None):
            if datos:
                self.guardar_caracteristicas(datos, csv_path)
            if video_writer is not None and frame_anotado is not None:
                video_writer.write(frame_anotado)

        if usar_threads:
            cola_frames = Queue(maxsize=30)
            cola_resultados = Queue()
            evento_terminar = threading.Event()

            # Función de procesamiento en hilo separado
            def procesar_cola():
                while True:
                    try:
                        idx, frame = cola_frames.get(timeout=1.0)
                    except Empty:
                        if evento_terminar.is_set():
                            return
                        continue

                    frame_anotado, datos = self.procesar_frame(frame, dibujar=mostrar)
                    cola_resultados.put((idx, frame_anotado, datos))
                    cola_frames.task_done()

            # Iniciar hilo de procesamiento
            hilo_proc = threading.Thread(target=procesar_cola)
            hilo_proc.daemon = True
            hilo_proc.start()

        print(f"Procesando video: {video_path}")

        # Índice local usado solo para encolar en orden; el contador real de
        # frames (self.frame_num) lo mantiene procesar_frame.
        indice_envio = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if usar_threads:
                # Poner frame en cola para procesar
                cola_frames.put((indice_envio, frame))
                indice_envio += 1

                # Recoger todos los resultados que ya estén listos (antes solo
                # se revisaba una vez por iteración, y el resto se perdía sin
                # guardarse en el CSV)
                while True:
                    try:
                        _, frame_anotado, datos = cola_resultados.get_nowait()
                    except Empty:
                        break
                    guardar_resultado(datos, frame_anotado)

                    # # Mostrar si es necesario
                    # if mostrar:
                    #     cv2.imshow("Análisis de Video", frame_anotado)
                    #     if cv2.waitKey(1) & 0xFF == ord('q'):
                    #         break
            else:
                # Procesamiento secuencial
                frame_anotado, datos = self.procesar_frame(frame, dibujar=mostrar)
                guardar_resultado(datos, frame_anotado)

                # # Mostrar si es necesario
                # if mostrar:
                #     cv2.imshow("Análisis de Video", frame_anotado)
                #     if cv2.waitKey(1) & 0xFF == ord('q'):
                #         break

        # Limpiar
        if usar_threads:
            # Esperar a que el hilo procese todo lo que quede en la cola
            cola_frames.join()
            evento_terminar.set()
            hilo_proc.join()

            # Drenar cualquier resultado que haya quedado pendiente de guardar
            while True:
                try:
                    _, frame_anotado, datos = cola_resultados.get_nowait()
                except Empty:
                    break
                guardar_resultado(datos, frame_anotado)

        cap.release()
        if video_writer is not None:
            video_writer.release()
        if mostrar:
            cv2.destroyAllWindows()

        print(f"Procesamiento completado. Resultados guardados en {csv_path}")
        if guardar_video:
            print(f"Video anotado guardado en {guardar_video}")

# Ejemplo de uso
if __name__ == "__main__":
    detector = ObjectDetector()

    import glob

    # Directorio raíz del proyecto (este script vive en code/)
    directorio_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Los CSV se guardan en un directorio plano con el nombre prefijado por la
    # etiqueta ("normal_" / "sospechoso_"), que es la convención que espera
    # VideoFrameDataset en model.py para asignar la clase de cada video.
    for etiqueta in ("normal", "sospechoso"):
        patron_videos = os.path.join(directorio_base, "dataset", etiqueta, "*.mp4")
        for video in glob.glob(patron_videos):
            nombre = os.path.basename(video).split('.')[0]
            csv_path = os.path.join(
                directorio_base, "informacion", "csv", f"{etiqueta}_{nombre}.csv"
            )
            detector.procesar_video(video, csv_path)
    