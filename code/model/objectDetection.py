# Importar librerías
from ultralytics import YOLO 
import cv2                 
import math                  
import featureExtraction as fe  # Importar el módulo featureExtraction para las caracteristicas
import numpy as np            
import csv
import os

""" 
Notas:
    * Darle mas entrenamiento al modelo ya que no detecta bien a personas volteadas o desde el angulo arriba
    * Da muchos falsos positivos, ve personas donde no las hay
    
    * El archivo de excel con las caracteristicas es demasiado grande, se podria reducir saltando algunos frames
    
    * El entrenamiento de manos se realizó con 10 epocas
"""

modelo_personas = YOLO("yolo-Weights/yolov8n.pt")  # Modelo entrenado para personas
modelo_manos = YOLO("./runs/detect/train4/weights/best.pt")  # Modelo entrenado para manos


# Función para las bounding boxes y centroides
def procesar_detecciones(resultados, color, etiqueta,img,centroides_actuales):
    for r in resultados:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confianza = math.ceil((box.conf[0] * 100)) / 100
            
            cx, cy = fe.calcular_centroide(x1, y1, x2, y2)  # Calcular el centroide
            
            print(f"{etiqueta} - Centroide: ({cx}, {cy}) - Confianza: {confianza}")
            
            key = f"{etiqueta}_{len(centroides_actuales)}"
            centroides_actuales[key] = (cx, cy)
            
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            cv2.putText(img, etiqueta, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        1, color, 2)
            
            cv2.circle(img, (cx, cy), 5, color, -1)
            
            
# Función principal donde se procesa el video
def modelo(video_path): 
    centroides_anteriores = {}
    velocidades_anteriores = {}
    datos_frame = []

    cap = cv2.VideoCapture(video_path)

    frame_anterior = None
    frame_num = 0  # Contador de frames

    # Obtener FPS del video
    fps = cap.get(cv2.CAP_PROP_FPS)

    while True:
        success, img = cap.read()
        if not success:
            break

        frame_actual = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        centroides_actuales = {}
        
        # Detección con ambos modelos
        resultados_personas = modelo_personas(img, stream=True)
        resultados_manos = modelo_manos(img, stream=True)

        # Dibujar boundingboxes en la imagen
        procesar_detecciones(resultados_personas, (255, 0, 0), "Persona", img, centroides_actuales)
        procesar_detecciones(resultados_manos, (0, 255, 0), "Mano", img, centroides_actuales)
        
        # Calcular flujo óptico de los centroides
        centroides_actualizados = fe.flujoOptico(centroides_actuales, frame_anterior, frame_actual)

        frame_dimensiones = img.shape[:2]  # Obtener alto y ancho del frame

        # Calcular desplazamiento, velocidad, aceleracion
        desplazamientos = fe.desplazamientoPixeles(centroides_anteriores, centroides_actualizados)
        velocidades = fe.velocidadDesplazamiento(centroides_anteriores, centroides_actualizados, fps)
        aceleraciones = fe.aceleracionDesplazamiento(velocidades_anteriores, velocidades, fps)
        densidad = fe.densidadMovimiento(centroides_actualizados, frame_dimensiones)

        # Imprimir los desplazamientos y velocidades
        for key in desplazamientos:
            print(f"{key} - Desplazamiento: {desplazamientos[key]:.2f} píxeles")
        for key in velocidades:
            print(f"{key} - Velocidad: {velocidades[key]:.2f} píxeles/seg")
        for key in aceleraciones:
            print(f"{key} - Aceleración: {aceleraciones[key]:.2f} píxeles/seg²")

        print(f"Densidad de movimiento: {densidad:.6f}")

        for key, (cx, cy) in centroides_actualizados.items():
            if key in desplazamientos and key in velocidades and key in aceleraciones:
                datos_frame.append({
                    "Frame": frame_num,
                    "Objeto": key,
                    "Centroide_X": cx,
                    "Centroide_Y": cy,
                    "Desplazamiento": desplazamientos[key],
                    "Velocidad": velocidades[key],
                    "Aceleración": aceleraciones[key],
                    "Densidad": densidad
                })
            else:
                print(f"Advertencia: La clave {key} no está en los diccionarios de desplazamientos, velocidades o aceleraciones.")


        # Guardar los datos del frame actual en el CSV
        fe.guardar_datos_csv("datos_movimiento.csv", datos_frame)

        # Mostrar el flujo óptico y el movimiento en la imagen
        for key, (cx, cy) in centroides_actualizados.items():
            if key in centroides_anteriores:
                cx_ant, cy_ant = centroides_anteriores[key]
                cv2.arrowedLine(img, (cx_ant, cy_ant), (cx, cy), (0, 255, 255), 2)

        # Actualizar datos
        frame_anterior = frame_actual.copy()
        velocidades_anteriores = velocidades.copy()
        centroides_anteriores = centroides_actualizados.copy()

        frame_num += 1  # Aumentar contador de frames

        cv2.imshow('Video', img)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Ruta del archivo de video
video_path = "./dataset/sospechoso/3.mp4"

# Llamar a la función principal           
modelo(video_path)