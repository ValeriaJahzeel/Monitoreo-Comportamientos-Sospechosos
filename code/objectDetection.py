"""
Módulo de detección de objetos para análisis de movimientos sospechosos.
Versión optimizada que aprovecha FeatureExtractor.
"""
import cv2
import numpy as np
import os
from ultralytics import YOLO
import featureExtraction as fe
# from sort.sort import Sort

# En la inicialización
# self.tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)

class ObjectDetector:
    def __init__(self, model_path="yolo-Weights/yolov8n.pt"):
        # Inicializar modelo YOLO
        self.modelo = YOLO(model_path)
        # Inicializar extractor de características
        self.feature_extractor = fe.FeatureExtractor(history_size=30)
        # Variables para seguimiento
        self.frame_anterior = None
        self.frame_num = 0
    
    def detectar_objetos(self, frame):
        """Detecta objetos usando YOLO y retorna los bounding boxes"""
        resultados = self.modelo(frame, stream=True)
        bboxes = {}
        
        for r in resultados:
            for i, box in enumerate(r.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                # Usar ID único basado en el frame actual y contador
                obj_id = f"{self.frame_num}_{i}"
                bboxes[obj_id] = (x1, y1, x2, y2)
        
        return bboxes
    
    # def detectar_objetos(self, frame):
    #     resultados = self.modelo(frame, stream=True)
    #     detecciones = []
        
    #     for r in resultados:
    #         for box in r.boxes:
    #             x1, y1, x2, y2 = map(int, box.xyxy[0])
    #             conf = float(box.conf[0])
    #             cls = int(box.cls[0])
    #             detecciones.append([x1, y1, x2, y2, conf])
        
    #     # Formato para SORT: [x1,y1,x2,y2,conf]
    #     if detecciones:
    #         tracked_objects = self.tracker.update(np.array(detecciones))
    #         bboxes = {}
    #         for track in tracked_objects:
    #             x1, y1, x2, y2, id_obj = track[:5]
    #             bboxes[f"obj_{int(id_obj)}"] = (int(x1), int(y1), int(x2), int(y2))
    #         return bboxes
    #     return {}
    
    def determinar_roi_flujo(self, flujo, umbral=1.0, densidad_min=10):
        """Determina una región de interés basada en el flujo óptico"""
        # Calcular magnitud del flujo
        magnitud, _ = cv2.cartToPolar(flujo[..., 0], flujo[..., 1])
        # Filtrar por umbral
        confianza = magnitud > umbral
        densidad = np.sum(confianza)
        
        # Si no hay suficiente flujo, usar todo el cuadro
        if densidad < densidad_min:
            return 0, 0, flujo.shape[1], flujo.shape[0]
        
        # Calcular ROI basada en áreas con flujo intenso
        roi = np.where(confianza)
        if len(roi[0]) == 0 or len(roi[1]) == 0:
            return 0, 0, flujo.shape[1], flujo.shape[0]
            
        x_min, x_max = np.min(roi[1]), np.max(roi[1])
        y_min, y_max = np.min(roi[0]), np.max(roi[0])
        
        return x_min, y_min, x_max, y_max
    
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
            x_min, y_min, x_max, y_max = self.determinar_roi_flujo(flujo_data)
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
    
    def procesar_video(self, video_path, csv_path, mostrar=True):
        """Procesa un video completo y guarda los resultados"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: No se pudo abrir el video {video_path}")
            return
        
        # Resetear contador de frames y otras variables
        self.frame_num = 0
        self.frame_anterior = None
        
        # Obtener información del video
        fps = cap.get(cv2.CAP_PROP_FPS)
        ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Configurar escritor de video (opcional)
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # ruta_salida = os.path.join(os.path.dirname(csv_path), 
        #                          os.path.basename(video_path).split('.')[0] + '_analizado.mp4')
        # out = cv2.VideoWriter(ruta_salida, fourcc, fps, (ancho, alto))
        
        print(f"Procesando video: {video_path}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Procesar frame actual
            frame_anotado, datos = self.procesar_frame(frame)
            
            # Guardar datos de características
            if datos:
                self.guardar_caracteristicas(datos, csv_path)
            
            # Escribir frame procesado en el video de salida
            # out.write(frame_anotado)
            
            # Mostrar progreso
            if mostrar:
                cv2.imshow("Análisis de Video", frame_anotado)
                if cv2.waitKey(1) & 0xFF == ord('q'): 
                    break
        
        # Liberar recursos
        cap.release()
        # out.release()
        if mostrar:
            cv2.destroyAllWindows()
        
        print(f"Procesamiento completado. Resultados guardados en {csv_path}")
        # print(f"Video con anotaciones guardado como {ruta_salida}")

# Ejemplo de uso
if __name__ == "__main__":
    detector = ObjectDetector()
    
    # Procesar un solo video
    video_path = r'D:\Documentos\Monitoreo-Comportamientos-Sospechosos\dataset\sospechoso\5.mp4'
    csv_path = "./informacion/csv/sospechoso/5_2.csv"
    detector.procesar_video(video_path, csv_path)
    
    # Para procesar múltiples videos
    
    # import glob
    # videos = glob.glob("./dataset/sospechoso/*.mp4")
    # for video in videos:
    #     nombre = os.path.basename(video).split('.')[0]
    #     csv_path = f"./informacion/csv/sospechoso/{nombre}.csv"
    #     detector.procesar_video(video, csv_path)
    