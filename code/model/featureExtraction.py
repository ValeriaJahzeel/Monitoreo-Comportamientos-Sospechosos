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


def flujoOptico(frame_anterior, frame_actual, bboxes):
    centroides_actualizados = {}
    
    if frame_anterior is None:
        return centroides_actualizados
        
    for key, bbox in bboxes.items():
        x1, y1, x2, y2 = bbox
        
        # Extraer la región de interés (ROI) de ambos frames
        roi_anterior = frame_anterior[y1:y2, x1:x2]
        roi_actual = frame_actual[y1:y2, x1:x2]
        
        # Verificar que las ROIs no estén vacías
        if roi_anterior.size == 0 or roi_actual.size == 0:
            continue
            
        # Calcular el flujo óptico solo en la ROI
        flow = cv2.calcOpticalFlowFarneback(
            roi_anterior, 
            roi_actual,
            None,
            0.5,  # pyr_scale
            3,    # levels
            15,   # winsize
            3,    # iterations
            5,    # poly_n
            1.2,  # poly_sigma
            0     # flags
        )
        
        # Calcular el desplazamiento promedio dentro del bbox
        mean_flow = np.mean(flow, axis=(0, 1))
        
        # Calcular el centroide del bbox
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        
        # Actualizar la posición del centroide según el flujo óptico
        new_cx = int(cx + mean_flow[0])
        new_cy = int(cy + mean_flow[1])
        
        centroides_actualizados[key] = (new_cx, new_cy)
    
    return centroides_actualizados

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



