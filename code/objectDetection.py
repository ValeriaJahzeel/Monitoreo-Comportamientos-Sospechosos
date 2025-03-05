# Importar librerías
from ultralytics import YOLO 
import cv2                 
import math                  
import featureExtraction as fe  # Importar el módulo featureExtraction para las características
import numpy as np            
import csv
import os

# Modelos YOLO
modelo_personas = YOLO("yolo-Weights/yolov8n.pt")  # Modelo entrenado para personas
modelo_manos = YOLO("./runs/detect/train4/weights/best.pt")  # Modelo entrenado para manos


# Función para las bounding boxes y centroides
def procesar_detecciones(resultados, color, etiqueta, img, centroides_actuales, bbox_actuales):
    for r in resultados:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confianza = math.ceil((box.conf[0] * 100)) / 100

            if confianza > 0.7:         # Si la confianza es mayor a 0.7
                cx, cy = fe.calcular_centroide(x1, y1, x2, y2)

                print(f"{etiqueta} - Centroide: ({cx}, {cy}) - Confianza: {confianza}")

                key = f"{etiqueta}_{len(centroides_actuales)}"
                centroides_actuales[key] = (cx, cy)
                bbox_actuales[key] = (x1, y1, x2, y2)

                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                cv2.putText(img, etiqueta, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)


# Función principal donde se procesa el video
def modelo(video_path): 
    centroides_anteriores = {}
    velocidades_anteriores = {}
    datos_frame = []
    objetos_en_area = {}  # Diccionario para contar los frames que un objeto está en un área

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    ret, primer_frame = cap.read()
    if not ret:
        return
        
    frame_anterior = cv2.cvtColor(primer_frame, cv2.COLOR_BGR2GRAY)
    frame_num = 0

    while True:
        success, img = cap.read()
        if not success:
            break

        frame_actual = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        centroides_actuales = {}
        bbox_actuales = {}

        # Detección con ambos modelos
        resultados_personas = modelo_personas(img, stream=True)
        resultados_manos = modelo_manos(img, stream=True)

        # Dibujar boundingboxes en la imagen
        procesar_detecciones(resultados_personas, (255, 0, 0), "Persona", img, centroides_actuales, bbox_actuales)
        procesar_detecciones(resultados_manos, (0, 255, 0), "Mano", img, centroides_actuales, bbox_actuales)
        
        # Calcular flujo óptico de los centroides
        flujos_por_bbox, _ = fe.flujoOptico(frame_anterior, frame_actual, bbox_actuales)
        img = fe.visualizar_flujo(img, flujos_por_bbox, bbox_actuales)

        frame_dimensiones = img.shape[:2]

        # Calcular métricas
        desplazamientos = fe.desplazamientoPixeles(centroides_anteriores, centroides_actuales)
        velocidades = fe.velocidadDesplazamiento(centroides_anteriores, centroides_actuales, fps)
        aceleraciones = fe.aceleracionDesplazamiento(velocidades_anteriores, velocidades, fps)
        densidad = fe.densidadMovimiento(centroides_actuales, frame_dimensiones)
        direcciones = fe.direccionMovimiento(centroides_anteriores, centroides_actuales)
        posturas = fe.deteccionPostura(bbox_actuales)

        # Calcular tiempo de permanencia y permanencia en área
        for key in centroides_actuales:
            if key in objetos_en_area:
                objetos_en_area[key] += 1
            else:
                objetos_en_area[key] = 1

        tiempos_permanencia = fe.tiempoPermanencia(objetos_en_area, fps)
        permanencia_area = fe.permanenciaArea(objetos_en_area, fps)

        # Imprimir métricas
        for key in desplazamientos:
            print(f"{key} - Desplazamiento: {desplazamientos[key]:.2f} píxeles")
        for key in velocidades:
            print(f"{key} - Velocidad: {velocidades[key]:.2f} píxeles/seg")
        for key in aceleraciones:
            print(f"{key} - Aceleración: {aceleraciones[key]:.2f} píxeles/seg²")
        for key in direcciones:
            print(f"{key} - Dirección: {direcciones[key]:.2f} grados")
        for key in posturas:
            print(f"{key} - Postura: {posturas[key]}")
        for key in tiempos_permanencia:
            print(f"{key} - Tiempo Permanencia: {tiempos_permanencia[key]:.2f} seg")
        for key in permanencia_area:
            print(f"{key} - Permanencia Área: {permanencia_area[key]:.2f} seg")

        print(f"Densidad de movimiento: {densidad:.6f}")

        # Guardar datos del frame
        for key, (cx, cy) in centroides_actuales.items():
            if key in desplazamientos and key in velocidades and key in aceleraciones:
                datos_frame.append({
                    "Frame": frame_num,
                    "Objeto": key,
                    "Centroide_X": cx,
                    "Centroide_Y": cy,
                    "Desplazamiento": desplazamientos[key],
                    "Velocidad": velocidades[key],
                    "Aceleracion": aceleraciones[key],
                    "Densidad": densidad,
                    "Direccion": direcciones.get(key, None),
                    "Tiempo_Permanencia": tiempos_permanencia.get(key, None),
                    "Postura": posturas.get(key, None),
                    "Permanencia_Area": permanencia_area.get(key, None)
                })

        # Guardar datos en CSV
        fe.guardar_datos_csv("archivooooo.csv", datos_frame)

        # Mostrar frame
        cv2.imshow('Video', img)
        
        # Actualizar datos para el siguiente frame
        frame_anterior = frame_actual.copy()
        velocidades_anteriores = velocidades.copy()
        centroides_anteriores = centroides_actuales.copy()
        frame_num += 1

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Ruta del archivo de video
video_path = "./videos_mejorados/1_enhanced.mp4"

# Llamar a la función principal           
modelo(video_path)