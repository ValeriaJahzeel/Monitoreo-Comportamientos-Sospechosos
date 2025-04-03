'''
Este script cambia el nombre de los archivos .csv
'''

import cv2
import os

# Rutas
input_path = r'D:\Documentos\Monitoreo-Comportamientos-Sospechosos\informacion\csv\normal'
output_path = r'D:\Documentos\Monitoreo-Comportamientos-Sospechosos\datasetCSV'

# Crear el directorio de salida si no existe
os.makedirs(output_path, exist_ok=True)

with os.scandir(input_path) as archivos:
    for archivo in archivos:
        if archivo.is_file():  # Asegurarse de que sea un archivo
            print(f"Procesando: {archivo.name}")
            
            # Ruta completa del archivo de entrada y salida
            input_file = os.path.join(input_path, archivo.name)
            new_filename = f"normal_{archivo.name}"
            output_file = os.path.join(output_path, new_filename)
            
            # Renombrar y mover el archivo
            os.rename(input_file, output_file)
        #  # Ruta completa del video
        # input_file = os.path.join(input_path, archivo.name)
        # output_file = os.path.join(output_path, f"normal_{archivo.name}")
        

