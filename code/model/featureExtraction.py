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

def direccionMovimiento():
    pass

def tiempoPermanencia():
    pass

def deteccionPostura():
    pass


# Calcula el flujo optico a partir de los centroides de los objetos detectados
centroides_anteriores = {}  
def flujoOptico(centroides_actuales, frame_anterior, frame_actual):
    global centroides_anteriores

    # Parámetros de Lucas-Kanade
    lk_params = dict(winSize=(15, 15), maxLevel=2, 
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    if frame_anterior is None or not centroides_anteriores:
        centroides_anteriores = centroides_actuales.copy()
        return centroides_actuales  # No hay flujo óptico en el primer frame

    # solo los centroides se calculan
    puntos_anteriores = np.array(list(centroides_anteriores.values()), dtype=np.float32).reshape(-1, 1, 2)
    
    # flujo óptico SOLO en los centroides
    puntos_actuales, status, _ = cv2.calcOpticalFlowPyrLK(frame_anterior, frame_actual, puntos_anteriores, None, **lk_params)

    if puntos_actuales is not None:
        for i, key in enumerate(centroides_anteriores.keys()):
            if status[i]:
                cx, cy = puntos_actuales[i].ravel()
                centroides_actuales[key] = (int(cx), int(cy))  

    centroides_anteriores = centroides_actuales.copy()
    return centroides_actuales


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


def permanenciaArea():
    pass

# Guarda los datos de cada frame en un archivo CSV
def guardar_datos_csv(nombre_archivo, datos):
    encabezados = ["Frame", "Objeto", "Centroide_X", "Centroide_Y", "Desplazamiento", "Velocidad", "Aceleración", "Densidad"]

    # Si el archivo no existe, escribir encabezados
    archivo_nuevo = not os.path.exists(nombre_archivo)
    
    with open(nombre_archivo, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=encabezados)

        if archivo_nuevo:
            writer.writeheader()  # Escribir encabezado solo si el archivo es nuevo

        writer.writerows(datos)  # Escribir los datos



