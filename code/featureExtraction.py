"""
Módulo de extracción de características para análisis de movimientos sospechosos.
Versión optimizada con funciones adicionales.
"""
import cv2
import math
import numpy as np
import os
import csv
from collections import deque
from scipy.spatial.distance import cdist

class FeatureExtractor:
    def __init__(self, history_size=30):
        self.history_size = history_size
        self.trayectorias = {}  # Almacena trayectorias históricas
        self.centroides_anteriores = {}
        self.velocidades_anteriores = {}
        self.frame_dimensiones = None
    
    def update_frame_dimensions(self, dimensions):
        """Actualiza las dimensiones del frame actual para cálculos relativos"""
        self.frame_dimensiones = dimensions
    
    def calcular_centroide(self, bbox):
        """Calcula el centroide de un bounding box (x1, y1, x2, y2)"""
        x1, y1, x2, y2 = bbox
        return (x1 + x2) // 2, (y1 + y2) // 2
    
    def actualizar_centroides(self, bboxes):
        """Actualiza centroides a partir de los bounding boxes detectados"""
        centroides = {}
        for key, bbox in bboxes.items():
            centroides[key] = self.calcular_centroide(bbox)
            
            # Inicializar o actualizar historial de trayectoria
            if key not in self.trayectorias:
                self.trayectorias[key] = deque(maxlen=self.history_size)
            
            self.trayectorias[key].append(centroides[key])
        
        # Guardar para el siguiente frame (no sobreescribir resultado)
        resultado = centroides.copy()  # Usar copy para evitar referencias cruzadas
        
        # Actualizar centroides anteriores para el próximo frame
        self.centroides_anteriores = self.centroides_anteriores.copy()  # Preservar los anteriores
        
        return resultado
    
    def calcular_metricas_movimiento(self, centroides, fps=25):
        """
        Calcula múltiples métricas de movimiento en un solo paso:
        - Desplazamiento en píxeles
        - Velocidad (píxeles/seg)
        - Aceleración (píxeles/seg²)
        - Dirección (ángulo en grados)
        
        Devuelve un diccionario con todas las métricas.
        """
        result = {
            'desplazamiento': {},
            'velocidad': {},
            'aceleracion': {},
            'direccion': {}
        }
        
        if not self.centroides_anteriores or not centroides or fps <= 0:
            return result
        
        for key in centroides:
            if key in self.centroides_anteriores:
                x1, y1 = self.centroides_anteriores[key]
                x2, y2 = centroides[key]
                
                # Cálculo vectorizado
                dx, dy = x2 - x1, y2 - y1
                desplazamiento = np.sqrt(dx**2 + dy**2)
                velocidad = desplazamiento * fps
                
                # Dirección del movimiento (ángulo)
                angulo = math.degrees(math.atan2(dy, dx))
                
                # Almacenar resultados
                result['desplazamiento'][key] = desplazamiento
                result['velocidad'][key] = velocidad
                result['direccion'][key] = angulo
                
                # Calcular aceleración si hay datos anteriores
                if key in self.velocidades_anteriores:
                    v1 = self.velocidades_anteriores[key]
                    result['aceleracion'][key] = (velocidad - v1) * fps
        
        # Actualizar velocidades para el próximo cálculo
        self.velocidades_anteriores = result['velocidad'].copy()
        
        return result
    
    def detectar_postura(self, bboxes):
        """Detecta la postura basada en la relación de aspecto del bounding box"""
        posturas = {}
        
        for key, bbox in bboxes.items():
            x1, y1, x2, y2 = bbox
            ancho = x2 - x1
            alto = y2 - y1
            
            if ancho <= 0 or alto <= 0:
                posturas[key] = "Desconocida"
                continue
                
            relacion = ancho / alto
            
            if relacion > 1.5:
                posturas[key] = "Horizontal"
            elif relacion < 0.5:
                posturas[key] = "Vertical"
            else:
                posturas[key] = "Neutral"
                
        return posturas
    
    def calcular_densidad_movimiento(self, centroides):
        """Calcula la densidad de movimiento normalizada en el frame"""
        if not centroides or not self.frame_dimensiones:
            return 0.0
            
        alto, ancho = self.frame_dimensiones
        area_total = alto * ancho
        
        # Usar vectorización para sumar todos los desplazamientos
        sum_movement = sum(np.linalg.norm(np.array(centroides[k])) for k in centroides)
        return sum_movement / area_total
    
    def calcular_flujo_optico(self, frame_anterior, frame_actual, bboxes, method='sparse'):
        """
        Calcula el flujo óptico usando método disperso (Lucas-Kanade) o denso (Farneback)
        """
        if frame_anterior is None:
            return {}, {}
            
        if method == 'dense':
            # Flujo óptico denso con Farneback
            return cv2.calcOpticalFlowFarneback(
                frame_anterior, frame_actual, 
                None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
        else:
            # Flujo óptico disperso con Lucas-Kanade
            flujos_por_bbox = {}
            puntos_por_bbox = {}
            
            # Parámetros LK
            lk_params = dict(
                winSize=(15, 15),
                maxLevel=4,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
            )
            
            # Parámetros para puntos característicos
            feature_params = dict(
                maxCorners=100,
                qualityLevel=0.1,
                minDistance=5,
                blockSize=7
            )
            
            for key, bbox in bboxes.items():
                x1, y1, x2, y2 = bbox
                roi = frame_anterior[y1:y2, x1:x2]
                
                if roi.size == 0:
                    continue
                    
                # Detectar puntos característicos en ROI
                puntos = cv2.goodFeaturesToTrack(roi, mask=None, **feature_params)
                
                if puntos is not None and len(puntos) > 0:
                    # Ajustar coordenadas al frame completo
                    puntos[:, :, 0] += x1
                    puntos[:, :, 1] += y1
                    
                    # Calcular flujo LK
                    nuevos_puntos, status, _ = cv2.calcOpticalFlowPyrLK(
                        frame_anterior, frame_actual, puntos, None, **lk_params
                    )
                    
                    # Filtrar buenos puntos
                    buenos_idx = status.ravel() == 1
                    buenos_nuevos = nuevos_puntos[buenos_idx]
                    buenos_viejos = puntos[buenos_idx]
                    
                    if len(buenos_nuevos) > 0:
                        flujos_por_bbox[key] = (buenos_viejos, buenos_nuevos)
                        puntos_por_bbox[key] = buenos_nuevos
            
            return flujos_por_bbox, puntos_por_bbox
    
    def visualizar_flujo(self, img, flujo_data, bboxes=None, method='sparse'):
        """Visualiza el flujo óptico en la imagen"""
        if method == 'dense':
            # Visualización de flujo denso
            h, w = img.shape[:2]
            paso = 16
            y, x = np.mgrid[0:h:paso, 0:w:paso].reshape(2, -1).astype(int)
            fx, fy = flujo_data[y, x].T
            
            # Crear una copia para no modificar la original
            vis_img = img.copy()
            
            # Dibujar flechas
            lineas = np.vstack([x, y, x + fx * 3, y + fy * 3]).T.reshape(-1, 2, 2)
            lineas = np.int32(lineas + 0.5)
            
            for (x1, y1), (x2, y2) in lineas:
                cv2.arrowedLine(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 1, tipLength=0.3)
                
            return vis_img
        else:
            # Visualización de flujo disperso
            mask = np.zeros_like(img)
            
            for key, (puntos_viejos, puntos_nuevos) in flujo_data.items():
                if bboxes and key in bboxes:
                    x1, y1, x2, y2 = bboxes[key]
                    
                    for i, (new, old) in enumerate(zip(puntos_nuevos, puntos_viejos)):
                        a, b = new.ravel()
                        c, d = old.ravel()
                        
                        if x1 <= a <= x2 and y1 <= b <= y2:
                            # Color basado en ángulo
                            angle = np.arctan2(b - d, a - c) + np.pi
                            hue = angle / (2 * np.pi)
                            
                            color = np.uint8([[[hue * 180, 255, 255]]])
                            color = cv2.cvtColor(color, cv2.COLOR_HSV2BGR).ravel()
                            
                            cv2.line(
                                mask, (int(c), int(d)), (int(a), int(b)),
                                (int(color[0]), int(color[1]), int(color[2])), 2
                            )
                            cv2.circle(
                                mask, (int(a), int(b)), 1,
                                (int(color[0]), int(color[1]), int(color[2])), -1
                            )
            
            return cv2.addWeighted(img, 1, mask, 1, 0)
        
    def analizar_trayectoria(self, objeto_id):
        """
        Analiza la trayectoria de un objeto y detecta patrones de movimiento
        Devuelve métricas como: linealidad, circularidad, zigzag
        """
        if objeto_id not in self.trayectorias or len(self.trayectorias[objeto_id]) < 5:
            return {
                'linealidad': 0,
                'circularidad': 0,
                'zigzag': 0,
                'patron': 'Indeterminado'
            }
        
        # Convertir deque a array para análisis
        tray = np.array(self.trayectorias[objeto_id])
        
        # Calcular la distancia total recorrida
        dist_total = np.sum(np.sqrt(np.sum(np.diff(tray, axis=0)**2, axis=1)))
        
        # Calcular la distancia directa desde el inicio al fin
        dist_directa = np.sqrt(np.sum((tray[-1] - tray[0])**2))
        
        # Calcular linealidad (1 = perfectamente lineal)
        linealidad = dist_directa / (dist_total + 1e-6) if dist_total > 0 else 0
        
        # Calcular cambios bruscos de dirección (zigzag)
        if len(tray) > 2:
            vectores = np.diff(tray, axis=0)
            angulos = np.arccos(np.sum(vectores[:-1] * vectores[1:], axis=1) / 
                              (np.linalg.norm(vectores[:-1], axis=1) * np.linalg.norm(vectores[1:], axis=1) + 1e-6))
            zigzag = np.mean(np.abs(angulos)) / np.pi if len(angulos) > 0 else 0
        else:
            zigzag = 0
        
        # Calcular circularidad
        if len(tray) > 3:
            # Centro aproximado de la trayectoria
            centro = np.mean(tray, axis=0)
            # Distancias al centro
            distancias = np.sqrt(np.sum((tray - centro)**2, axis=1))
            # Variación de las distancias (menor = más circular)
            circularidad = 1 - (np.std(distancias) / (np.mean(distancias) + 1e-6))
        else:
            circularidad = 0
            
        # Determinar patrón predominante
        if linealidad > 0.8:
            patron = "Lineal"
        elif circularidad > 0.7:
            patron = "Circular"
        elif zigzag > 0.5:
            patron = "Zigzag"
        else:
            patron = "Mixto"
            
        return {
            'linealidad': linealidad,
            'circularidad': circularidad,
            'zigzag': zigzag,
            'patron': patron
        }
    
    def detectar_interacciones(self, centroides, umbral_distancia=50):
        """
        Detecta interacciones entre objetos basado en proximidad y movimiento relativo
        """
        interacciones = {}
        
        if len(centroides) < 2:
            return interacciones
            
        # Convertir diccionario a matrices para cálculo vectorizado
        obj_ids = list(centroides.keys())
        posiciones = np.array([centroides[k] for k in obj_ids])
        
        # Calcular matriz de distancias entre todos los pares
        dist_matrix = cdist(posiciones, posiciones)
        
        # Encontrar pares cercanos
        for i, id1 in enumerate(obj_ids):
            for j, id2 in enumerate(obj_ids):
                if i < j and dist_matrix[i, j] < umbral_distancia:  # Evitar duplicados y auto-interacciones
                    interacciones[(id1, id2)] = {
                        'distancia': dist_matrix[i, j],
                        'tipo': 'Proximidad' if dist_matrix[i, j] < umbral_distancia/2 else 'Cercana'
                    }
        
        return interacciones
    
    def detectar_comportamiento_ciclico(self, objeto_id, ventana_tiempo=20):
        """
        Detecta si un objeto está realizando movimientos repetitivos/cíclicos
        """
        if objeto_id not in self.trayectorias or len(self.trayectorias[objeto_id]) < ventana_tiempo:
            return {
                'es_ciclico': False,
                'frecuencia': 0,
                'amplitud': 0
            }
            
        # Tomar la ventana más reciente
        tray = np.array(list(self.trayectorias[objeto_id]))[-ventana_tiempo:]
        
        # Separar coordenadas x e y
        x = tray[:, 0]
        y = tray[:, 1]
        
        # Calcular autocorrelación para detectar patrones repetitivos
        autocorr_x = np.correlate(x - np.mean(x), x - np.mean(x), mode='full')
        autocorr_y = np.correlate(y - np.mean(y), y - np.mean(y), mode='full')
        
        # Normalizar
        autocorr_x = autocorr_x[len(autocorr_x)//2:] / autocorr_x[len(autocorr_x)//2]
        autocorr_y = autocorr_y[len(autocorr_y)//2:] / autocorr_y[len(autocorr_y)//2]
        
        # Buscar picos en la autocorrelación
        # (los picos indican periodicidad/ciclicidad)
        picos_x = (autocorr_x[1:-1] > autocorr_x[:-2]) & (autocorr_x[1:-1] > autocorr_x[2:])
        picos_y = (autocorr_y[1:-1] > autocorr_y[:-2]) & (autocorr_y[1:-1] > autocorr_y[2:])
        
        # Determinar si hay comportamiento cíclico
        hay_ciclo_x = np.any(picos_x) and np.max(autocorr_x[1:]) > 0.5
        hay_ciclo_y = np.any(picos_y) and np.max(autocorr_y[1:]) > 0.5
        
        es_ciclico = hay_ciclo_x or hay_ciclo_y
        
        # Calcular frecuencia y amplitud si es cíclico
        if es_ciclico:
            # Usar la coordenada con mayor ciclicidad
            if np.max(autocorr_x[1:]) > np.max(autocorr_y[1:]):
                indices_picos = np.where(picos_x)[0] + 1  # +1 por el desplazamiento del slice
                amplitud = np.std(x)
            else:
                indices_picos = np.where(picos_y)[0] + 1
                amplitud = np.std(y)
                
            if len(indices_picos) > 0:
                periodo = indices_picos[0]  # Frames entre repeticiones
                frecuencia = 1 / periodo if periodo > 0 else 0
            else:
                frecuencia = 0
                amplitud = 0
        else:
            frecuencia = 0
            amplitud = 0
            
        return {
            'es_ciclico': es_ciclico,
            'frecuencia': frecuencia,
            'amplitud': amplitud
        }
    
    def calcular_area_convex_hull(self, objeto_id):
        """
        Calcula el área del casco convexo de la trayectoria como medida de dispersión
        """
        if objeto_id not in self.trayectorias or len(self.trayectorias[objeto_id]) < 3:
            return 0
            
        # Convertir trayectoria a array numpy
        puntos = np.array(self.trayectorias[objeto_id])
        
        # Calcular casco convexo
        hull = cv2.convexHull(puntos.astype(np.float32))
        
        # Calcular área
        area = cv2.contourArea(hull)
        
        return area
    
    def recolectar_caracteristicas(self, frame_num, bboxes, fps=25):
        """
        Recolecta todas las características para el frame actual y objetos detectados
        """
        # Calcular centroides para este frame
        centroides = {}
        for key, bbox in bboxes.items():
            centroides[key] = self.calcular_centroide(bbox)
            
            # Inicializar o actualizar historial de trayectoria
            if key not in self.trayectorias:
                self.trayectorias[key] = deque(maxlen=self.history_size)
            
            self.trayectorias[key].append(centroides[key])
        
        # Calcular métricas básicas de movimiento con los centroides actuales y anteriores
        metricas = self.calcular_metricas_movimiento(centroides, fps)
        
        # Calcular posturas
        posturas = self.detectar_postura(bboxes)
        
        # Detectar interacciones
        interacciones = self.detectar_interacciones(centroides)
        
        # Recolectar características avanzadas por objeto
        datos_frame = []
        
        for obj_id in centroides:
            # Análisis de trayectoria
            trayectoria = self.analizar_trayectoria(obj_id)
            
            # Comportamiento cíclico
            ciclico = self.detectar_comportamiento_ciclico(obj_id)
            
            # Área de casco convexo (dispersión)
            area_hull = self.calcular_area_convex_hull(obj_id)
            
            # Combinar todas las características
            datos_obj = {
                "Frame": frame_num,
                "Objeto": obj_id,
                "Centroide_X": centroides[obj_id][0],
                "Centroide_Y": centroides[obj_id][1],
                "Desplazamiento": metricas['desplazamiento'].get(obj_id, 0),
                "Velocidad": metricas['velocidad'].get(obj_id, 0),
                "Aceleracion": metricas['aceleracion'].get(obj_id, 0),
                "Direccion": metricas['direccion'].get(obj_id, 0),
                "Densidad": self.calcular_densidad_movimiento(centroides),
                "Postura": posturas.get(obj_id, "Desconocida"),
                "Patron_Movimiento": trayectoria['patron'],
                "Linealidad": trayectoria['linealidad'],
                "Circularidad": trayectoria['circularidad'],
                "Zigzag": trayectoria['zigzag'],
                "Es_Ciclico": 1 if ciclico['es_ciclico'] else 0,
                "Frecuencia_Ciclo": ciclico['frecuencia'],
                "Amplitud_Ciclo": ciclico['amplitud'],
                "Area_Trayectoria": area_hull,
                "En_Interaccion": 1 if any(obj_id in par for par in interacciones) else 0
            }
            
            datos_frame.append(datos_obj)
        
        # Actualizar centroides anteriores DESPUÉS de calcular las métricas
        self.centroides_anteriores = centroides.copy()
        
        return datos_frame
    
    def guardar_datos_csv(self, nombre_archivo, datos):
        """Guarda los datos de características en un archivo CSV"""
        encabezados = [
            "Frame", "Objeto", "Centroide_X", "Centroide_Y", 
            "Desplazamiento", "Velocidad", "Aceleracion", "Direccion",
            "Densidad", "Postura", "Patron_Movimiento", "Linealidad",
            "Circularidad", "Zigzag", "Es_Ciclico", "Frecuencia_Ciclo",
            "Amplitud_Ciclo", "Area_Trayectoria", "En_Interaccion"
        ]
        
        # Verificar si el archivo ya existe
        archivo_nuevo = not os.path.exists(nombre_archivo)
        
        with open(nombre_archivo, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=encabezados)
            
            if archivo_nuevo:
                writer.writeheader()
                
            writer.writerows(datos)