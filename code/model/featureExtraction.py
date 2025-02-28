""" 
En esta sección se busca obtener diversas metricas de informacion que podria
resultar relevante para ayudar a la clasificación de movimientos sospechosos 
"""
import objectDetection as od
import cv2
import math
import numpy as np
import csv
import os


# Calcula el centroide de los bounding box
def calcular_centroide(x1, y1, x2, y2):  
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    return cx, cy


# Calcula el desplazamiento en pixeles de los centroides entre dos frames consecutivos
def desplazamientoPixeles(centroides_anteriores, centroides_actuales):
    desplazamientos = {}
    
    # Verificar si los centroides anteriores y actuales están vacíos
    if not centroides_anteriores or not centroides_actuales:
        print("Error: No hay centroides para comparar.")
        return desplazamientos
    
    for key in centroides_actuales:
        if key in centroides_anteriores:
            x1, y1 = centroides_anteriores[key]
            x2, y2 = centroides_actuales[key]
            
            # Calcular la distancia 
            desplazamiento = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            desplazamientos[key] = desplazamiento
            print(f"Desplazamiento - {key}: {desplazamiento:.2f} píxeles")

    return desplazamientos


# Calcula la velocidad de desplazamiento en píxeles/segundo.
def velocidadDesplazamiento(centroides_anteriores, centroides_actuales, fps):
    velocidades = {}

    if not centroides_anteriores or not centroides_actuales or fps == 0:
        print("Error: No hay datos suficientes o el FPS es cero.")
        return velocidades

    for key in centroides_actuales:
        if key in centroides_anteriores:
            x1, y1 = centroides_anteriores[key]
            x2, y2 = centroides_actuales[key]
            
            desplazamiento = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            velocidad = desplazamiento * fps  # Conversión a píxeles/segundo
            velocidades[key] = velocidad

    return velocidades


# Calcula la aceleración de desplazamiento en píxeles/seg².
def aceleracionDesplazamiento(velocidades_anteriores, velocidades_actuales, fps):
    aceleraciones = {}

    if not velocidades_anteriores or not velocidades_actuales or fps == 0:
        print("Error: No hay datos suficientes o el FPS es cero.")
        return aceleraciones

    for key in velocidades_actuales:
        if key in velocidades_anteriores:
            v1 = velocidades_anteriores[key]
            v2 = velocidades_actuales[key]

            aceleracion = (v2 - v1) * fps  # Derivada de la velocidad
            aceleraciones[key] = aceleracion

    return aceleraciones

def direccionMovimiento(centroides_anteriores, centroides_actuales):
    direcciones = {}

    if not centroides_anteriores or not centroides_actuales:
        print("Error: No hay datos de centroides para calcular la dirección.")
        return direcciones

    for key in centroides_actuales:
        if key in centroides_anteriores:
            x1, y1 = centroides_anteriores[key]
            x2, y2 = centroides_actuales[key]

            # Calcular el ángulo de la dirección en radianes
            dx = x2 - x1
            dy = y2 - y1
            angulo = math.atan2(dy, dx)  # Ángulo en radianes

            # Convertir a grados para facilitar la interpretación
            angulo_grados = math.degrees(angulo)
            direcciones[key] = angulo_grados

    return direcciones

def tiempoPermanencia(objetos_en_area, fps):
    tiempos = {}

    if not objetos_en_area or fps == 0:
        print("Error: No hay datos de objetos en el área o el FPS es cero.")
        return tiempos

    for key, frames in objetos_en_area.items():
        tiempo = frames / fps  # Convertir frames a segundos
        tiempos[key] = tiempo

    return tiempos

def deteccionPostura(bboxes):
    posturas = {}

    if not bboxes:
        print("Error: No hay bounding boxes para analizar.")
        return posturas

    for key, bbox in bboxes.items():
        x1, y1, x2, y2 = bbox
        ancho = x2 - x1
        alto = y2 - y1

        # Calcular la relación de aspecto (ancho/alto)
        relacion_aspecto = ancho / alto

        # Clasificar la postura basada en la relación de aspecto
        if relacion_aspecto > 1.5:
            posturas[key] = "Horizontal"
        elif relacion_aspecto < 0.5:
            posturas[key] = "Vertical"
        else:
            posturas[key] = "Neutral"

    return posturas


def visualizar_flujo(img, flujos_por_bbox, bboxes):
    """
    Visualiza el flujo óptico con líneas de colores que representan el movimiento
    """
    # Crear una máscara para dibujar las líneas de flujo
    mask = np.zeros_like(img)
    
    for key, (puntos_viejos, puntos_nuevos) in flujos_por_bbox.items():
        # Obtener la bbox actual
        x1, y1, x2, y2 = bboxes[key]
        
        # Dibujar las líneas de flujo
        for i, (new, old) in enumerate(zip(puntos_nuevos, puntos_viejos)):
            a, b = new.ravel()
            c, d = old.ravel()
            
            # Solo dibujar si el punto está dentro de la bbox
            if x1 <= a <= x2 and y1 <= b <= y2:
                # Calcular el ángulo del movimiento para determinar el color
                angle = np.arctan2(b - d, a - c) + np.pi
                hue = angle / (2 * np.pi)
                
                # Convertir HSV a BGR para obtener un color basado en la dirección
                color = np.uint8([[[hue * 180, 255, 255]]])
                color = cv2.cvtColor(color, cv2.COLOR_HSV2BGR).ravel()
                
                # Dibujar la línea con el color calculado
                cv2.line(mask, (int(c), int(d)), (int(a), int(b)), 
                        (int(color[0]), int(color[1]), int(color[2])), 2)
                
                # Dibujar un pequeño círculo en el punto final
                cv2.circle(mask, (int(a), int(b)), 1, 
                          (int(color[0]), int(color[1]), int(color[2])), -1)
    
    # Combinar la imagen original con la máscara de flujo óptico
    img_con_flujo = cv2.addWeighted(img, 1, mask, 1, 0)
    
    return img_con_flujo

def calcular_puntos_caracteristicos(frame, bbox):
    x1, y1, x2, y2 = bbox
    roi = frame[y1:y2, x1:x2]
    
    # Parámetros para detectar buenos puntos a seguir
    feature_params = dict(
        maxCorners=100,  # Aumentado para tener más puntos
        qualityLevel=0.1,  # Reducido para detectar más puntos
        minDistance=5,    # Reducido para permitir puntos más cercanos
        blockSize=7
    )
    
    # Detectar puntos característicos solo en la ROI
    puntos = cv2.goodFeaturesToTrack(roi, mask=None, **feature_params)
    
    if puntos is not None:
        # Ajustar las coordenadas de los puntos al frame completo
        puntos[:, :, 0] += x1
        puntos[:, :, 1] += y1
        return puntos
    return None

def flujoOptico(frame_anterior, frame_actual, bboxes):
    if frame_anterior is None:
        return {}, {}
    
    # Parámetros para Lucas-Kanade
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=4,       # Aumentado para mejor tracking
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )
    
    flujos_por_bbox = {}
    puntos_por_bbox = {}
    
    for key, bbox in bboxes.items():
        puntos = calcular_puntos_caracteristicos(frame_anterior, bbox)
        
        if puntos is not None and len(puntos) > 0:
            nuevos_puntos, status, error = cv2.calcOpticalFlowPyrLK(
                frame_anterior, 
                frame_actual, 
                puntos,
                None,
                **lk_params
            )
            
            # Filtrar solo los puntos buenos
            buenos_nuevos = nuevos_puntos[status == 1]
            buenos_viejos = puntos[status == 1]
            
            if len(buenos_nuevos) > 0:
                flujos_por_bbox[key] = (buenos_viejos, buenos_nuevos)
                puntos_por_bbox[key] = buenos_nuevos
    
    return flujos_por_bbox, puntos_por_bbox

# Calcula la densidad de movimiento en una región de interés (que tanto movimiento hay en un frame)
# El resultado está normalizado con respecto al área total del frame (de 0 a 1)
def densidadMovimiento(centroides_actuales, frame_dimensiones):
    if not centroides_actuales or not frame_dimensiones:
        print("Error: No hay datos de centroides o dimensiones del frame.")
        return 0.0

    alto, ancho = frame_dimensiones
    area_total = alto * ancho  # Área total del frame en píxeles

    # Calcular la cantidad de movimiento como la suma de desplazamientos
    total_movimiento = sum(np.linalg.norm(np.array(centroides_actuales[key])) for key in centroides_actuales)

    # Normalizar con respecto al área total
    densidad = total_movimiento / area_total

    return densidad


def permanenciaArea(objetos_en_area, fps):
    tiempos = {}

    if not objetos_en_area or fps == 0:
        print("Error: No hay datos de objetos en el área o el FPS es cero.")
        return tiempos

    for key, frames in objetos_en_area.items():
        tiempo = frames / fps  # Convertir frames a segundos
        tiempos[key] = tiempo

    return tiempos

# Guarda los datos de cada frame en un archivo CSV
def guardar_datos_csv(nombre_archivo, datos):
    
    # Definir los encabezados del archivo CSV
    encabezados = [
        "Frame", 
        "Objeto", 
        "Centroide_X", 
        "Centroide_Y", 
        "Desplazamiento", 
        "Velocidad", 
        "Aceleracion", 
        "Densidad", 
        "Direccion", 
        "Tiempo_Permanencia", 
        "Postura", 
        "Permanencia_Area"
    ]

    # Verificar si el archivo ya existe
    archivo_nuevo = not os.path.exists(nombre_archivo)
    
    # Abrir el archivo en modo de escritura (append)
    with open(nombre_archivo, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=encabezados)

        # Escribir los encabezados solo si el archivo es nuevo
        if archivo_nuevo:
            writer.writeheader()

        # Escribir los datos en el archivo CSV
        writer.writerows(datos)



video_path = "./videos_mejorados/1_enhanced.mp4"
