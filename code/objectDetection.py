"""
Módulo de detección de objetos para análisis de movimientos sospechosos.
Versión optimizada que aprovecha FeatureExtractor.
"""
import cv2
import numpy as np
import os
from ultralytics import YOLO
import featureExtraction as fe
import math
# from sort.sort import Sort

# En la inicialización
# self.tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)

class ObjectDetector:
    def __init__(self, model_path=r'D:\Documentos\Monitoreo-Comportamientos-Sospechosos\yolo-Weights\yolov8n.pt'):
        # Inicializar modelo YOLO
        self.modelo = YOLO(model_path)
        # Inicializar extractor de características
        self.feature_extractor = fe.FeatureExtractor(history_size=30)
        # Variables para seguimiento
        self.frame_anterior = None
        self.frame_num = 0
        self.objetos_previos = {}
        self.siguiente_id = 0
    
    def detectar_objetos(self, frame):
        """Detecta objetos usando YOLO y retorna los bounding boxes con IDs consistentes"""
        resultados = self.modelo(frame, stream=True)
        detecciones = []
        
        for r in resultados:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                # Calcular el centroide
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                detecciones.append((x1, y1, x2, y2, cx, cy, conf, cls))
        
        # Asignar IDs consistentes
        bboxes = {}
        
        # Establecer umbral de distancia para considerar que es el mismo objeto
        umbral_distancia = 50
        
        # Inicializar variables de seguimiento si no existen
        if not hasattr(self, 'objetos_previos'):
            self.objetos_previos = {}
            self.siguiente_id = 0
        
        # Inicializar conjuntos de asignación
        objetos_asignados = set()
        ids_asignados = set()
        
        # Matriz de costos para asociación - solo si hay objetos previos y detecciones actuales
        if self.objetos_previos and detecciones:
            objetos_anteriores = list(self.objetos_previos.items())
            
            # Calcular todas las distancias entre objetos previos y actuales
            matriz_distancias = np.zeros((len(objetos_anteriores), len(detecciones)))
            
            for i, (id_prev, (_, _, _, _, cx_prev, cy_prev)) in enumerate(objetos_anteriores):
                for j, (x1, y1, x2, y2, cx, cy, _, _) in enumerate(detecciones):
                    # Distancia euclídea entre centroides
                    dist = np.sqrt((cx - cx_prev)**2 + (cy - cy_prev)**2)
                    matriz_distancias[i, j] = dist
            
            # Ordenar todas las distancias de menor a mayor
            indices_planos = np.argsort(matriz_distancias.flatten())
            indices_2d = np.unravel_index(indices_planos, matriz_distancias.shape)
            
            for idx_prev, idx_det in zip(indices_2d[0], indices_2d[1]):
                # Si ya se asignó este objeto o detección, continuar
                if idx_prev in ids_asignados or idx_det in objetos_asignados:
                    continue
                    
                # Si la distancia es mayor que el umbral, no asignar
                if matriz_distancias[idx_prev, idx_det] > umbral_distancia:
                    continue
                    
                # Asignar ID
                id_obj = objetos_anteriores[idx_prev][0]
                x1, y1, x2, y2 = detecciones[idx_det][:4]
                bboxes[id_obj] = (x1, y1, x2, y2)
                
                # Marcar como asignados
                ids_asignados.add(idx_prev)
                objetos_asignados.add(idx_det)
        
        # Crear nuevos IDs para objetos no asignados
        for i, deteccion in enumerate(detecciones):
            if i not in objetos_asignados:
                id_nuevo = f"obj_{self.siguiente_id}"
                self.siguiente_id += 1
                x1, y1, x2, y2 = deteccion[:4]
                bboxes[id_nuevo] = (x1, y1, x2, y2)
        
        # Actualizar objetos previos para el siguiente frame
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
    
    def procesar_frame(self, frame):
        """Procesa un frame individual y devuelve resultados de análisis"""
        # Convertir a escala de grises para flujo óptico
        frame_gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.feature_extractor.update_frame_dimensions(frame.shape[:2])
        
        # Inicializar flujo y frame anotado
        flujo_data = None
        frame_anotado = frame.copy()
        
        # Calcular flujo óptico si hay frame anterior
        if self.frame_anterior is not None:
            # Calcular flujo óptico denso directamente
            flujo_data = self.calcular_flujo_optico_denso(self.frame_anterior, frame_gris)
        
        # Detectar objetos con YOLO
        bboxes = self.detectar_objetos(frame)
        
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
                    cv2.rectangle(frame_anotado, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.putText(frame_anotado, "Movimiento detectado", (x_min, y_min-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Recolectar todas las características disponibles
        datos_caracteristicas = self.feature_extractor.recolectar_caracteristicas(
            self.frame_num, bboxes, fps=25
        )
        
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
            frame_flujo = self.visualizar_flujo_denso(frame_anotado, flujo_data)
            frame_anotado = frame_flujo
        
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
    
    def procesar_video(self, video_path, csv_path, mostrar=True, usar_threads=True):
        """Procesa un video completo con soporte para multi-threading"""
        import threading
        from queue import Queue
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: No se pudo abrir el video {video_path}")
            return
        
        # Resetear variables
        self.frame_num = 0
        self.frame_anterior = None
        
        # Cola para procesamiento en paralelo
        if usar_threads:
            cola_frames = Queue(maxsize=30)
            cola_resultados = Queue()
            evento_terminar = threading.Event()
            
            # Función de procesamiento en hilo separado
            def procesar_cola():
                while not evento_terminar.is_set() or not cola_frames.empty():
                    try:
                        frame_data = cola_frames.get(timeout=1.0)
                        idx, frame = frame_data
                        
                        # Procesar frame
                        frame_anotado, datos = self.procesar_frame(frame)
                        
                        # Poner resultado en cola
                        cola_resultados.put((idx, frame_anotado, datos))
                        
                        # Marcar tarea como completada
                        cola_frames.task_done()
                    except:
                        continue
            
            # Iniciar hilo de procesamiento
            hilo_proc = threading.Thread(target=procesar_cola)
            hilo_proc.daemon = True
            hilo_proc.start()
        
        print(f"Procesando video: {video_path}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if usar_threads:
                # Poner frame en cola para procesar
                cola_frames.put((self.frame_num, frame))
                
                # Obtener y mostrar resultados procesados si están listos
                if not cola_resultados.empty():
                    idx, frame_anotado, datos = cola_resultados.get()
                    
                    # Guardar datos
                    if datos:
                        self.guardar_caracteristicas(datos, csv_path)
                    
                    # # Mostrar si es necesario
                    # if mostrar:
                    #     cv2.imshow("Análisis de Video", frame_anotado)
                    #     if cv2.waitKey(1) & 0xFF == ord('q'): 
                    #         break
            else:
                # Procesamiento secuencial
                frame_anotado, datos = self.procesar_frame(frame)
                
                # Guardar datos
                if datos:
                    self.guardar_caracteristicas(datos, csv_path)
                
                # # Mostrar si es necesario
                # if mostrar:
                #     cv2.imshow("Análisis de Video", frame_anotado)
                #     if cv2.waitKey(1) & 0xFF == ord('q'): 
                #         break
            
            self.frame_num += 1
        
        # Limpiar
        if usar_threads:
            evento_terminar.set()
            cola_frames.join()
            hilo_proc.join()
        
        cap.release()
        if mostrar:
            cv2.destroyAllWindows()
        
        print(f"Procesamiento completado. Resultados guardados en {csv_path}")

# Ejemplo de uso
if __name__ == "__main__":
    detector = ObjectDetector()
    
    # video_path = r'D:\Documentos\Monitoreo-Comportamientos-Sospechosos\dataset\sospechoso_reducido\1.mp4'
    # csv_path = "./informacion/csv/sospechoso/1.csv"
    # detector.procesar_video(video_path, csv_path)
    
    # varios videos
    
    import glob
    videos = glob.glob("./dataset/normal/*.mp4")
    for video in videos:
        nombre = os.path.basename(video).split('.')[0]
        csv_path = f"./informacion/csv/normal/{nombre}.csv"
        detector.procesar_video(video, csv_path)
    