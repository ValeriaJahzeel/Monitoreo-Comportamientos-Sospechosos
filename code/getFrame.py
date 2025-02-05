import cv2
import numpy as np
import os
import shutil
import mediapipe as mp

""" 
Extraer los frames de un video y guardarlos en una carpeta 
"""

def ObtenerFrames(rutaVideo, rutaCarpeta, rutaTrazos):
    # Eliminar carpeta si ya existe
    for carpeta in [rutaCarpeta, rutaTrazos]:
        if os.path.exists(carpeta):
            shutil.rmtree(carpeta)  # Elimina todo el contenido de la carpeta
            print(f"Se eliminó la carpeta: {carpeta}")
        
        os.makedirs(carpeta)  # Se vuelve a crear la carpeta vacía
        print(f"Carpeta creada: {carpeta}")

    cap = cv2.VideoCapture(rutaVideo)
    if not cap.isOpened():
        print("No se puede abrir el video.")
        return
    
    cont_frames = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Sale del bucle si no hay más frames
        
        if cont_frames % 10 == 0:  
            # Preprocesamiento: Ecualización de histograma
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convertir a escala de grises
            equalizado = cv2.equalizeHist(frame)
            
            # Guardar la imagen
            nombre_imagen = f'frame{cont_frames}.jpg'
            ruta_imagen = os.path.join(rutaCarpeta, nombre_imagen)
            resultado = cv2.imwrite(ruta_imagen, equalizado)
            
            if resultado:
                print(f"Guardado: {nombre_imagen}")
            else:
                print(f"Error al guardar {nombre_imagen}")

        cont_frames += 1

    cap.release()
    print("Extracción completada")

# Ejemplo de uso
# ObtenerFrames('./dataset/sospechoso/1.mp4', './frames/sospechoso/1', './trazos/sospechoso/1')
