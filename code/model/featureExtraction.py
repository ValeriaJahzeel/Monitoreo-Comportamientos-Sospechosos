""" 
En esta sección se busca obtener diversas metricas de informacion que podria
resultar relevante para ayudar a la clasificación de movimientos sospechosos 
"""
import objectDetection as od
import cv2
import math
import numpy as np

video_path = "./dataset/sospechoso/1.mp4"

def calcular_centroide(x1, y1, x2, y2):    # Calcula el centroide de los bounding box
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    return cx, cy

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
            
            # Calcular la distancia euclidiana en píxeles
            desplazamiento = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            desplazamientos[key] = desplazamiento
            print(f"Desplazamiento - {key}: {desplazamiento:.2f} píxeles")

    return desplazamientos


def velocidadDesplazamiento(centroides_anteriores, centroides_actuales, fps):
    velocidades = {}

    if fps == 0:
        return velocidades  # Evitar división por cero

    desplazamientos = desplazamientoPixeles(centroides_anteriores, centroides_actuales)

    for key, desplazamiento in desplazamientos.items():
        velocidades[key] = desplazamiento * fps  #Velocidad = distancia / tiempo (1/fps)

    return velocidades

def aceleracionDesplazamiento():
    pass

def direccionMovimiento():
    pass

def tiempoPermanencia():
    pass

def deteccionPostura():
    pass


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


def densidadMovimiento():
    pass

def permanenciaArea():
    pass

