import cv2
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
import shutil
import mediapipe as mp

# Extraer los frames y guardarlos en una carpeta
def ObtenerFrames(rutaVideo, rutaCarpeta, rutaTrazos):
    if not os.path.exists(rutaCarpeta):
        os.mkdir(rutaCarpeta)
        print("Carpeta creada corrrectamente" )
    else:
        print("Ya existe esa carpeta" )
        
        for filename in os.listdir(rutaCarpeta):
            file_path = os.path.join(rutaCarpeta, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('No se pudo eliminar')
                
    # para la carpeta de los trazos
    if not os.path.exists(rutaTrazos):
        os.mkdir(rutaTrazos)
        print("Carpeta creada corrrectamente" )
    else:
        print("Ya existe esa carpeta" )
        
        for filename in os.listdir(rutaTrazos):
            file_path = os.path.join(rutaTrazos, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('No se pudo eliminar')

    cap = cv2.VideoCapture(rutaVideo)
    if not cap.isOpened():
        print("No se puede abrir el video.")
        return
    
    cont_frames = 0
    
    while True:
        ret, frame = cap.read()
        
        if ret:
            if cont_frames % 10 == 0:  
                
                # preprocesamiento 1. equalizacion de histograma a las imagenes
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # primero se convierte a grises
                equalizado = cv2.equalizeHist(frame)
                
                
                # guardado de las imágenes
                name = 'frame' + str(cont_frames) + '.jpg'
                print(name)
                
                result = cv2.imwrite(os.path.join(rutaCarpeta, name), equalizado)
                if not result:
                    print(f"Error.")
            cont_frames += 1
        else:
            break
    
    cap.release()
    print("Extracción completada")