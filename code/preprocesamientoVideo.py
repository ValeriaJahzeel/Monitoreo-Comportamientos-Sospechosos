import cv2
import numpy as np
import os

def mejorar_video(ruta_entrada, carpeta_salida="videos_mejorados"):
    # Crear carpeta de salida si no existe
    if not os.path.exists(carpeta_salida):
        os.makedirs(carpeta_salida)
        print(f"Carpeta creada: {carpeta_salida}")
    
    # Generar ruta de salida
    nombre_archivo = os.path.basename(ruta_entrada)
    nombre_base = os.path.splitext(nombre_archivo)[0]
    ruta_salida = os.path.join(carpeta_salida, f"{nombre_base}_enhanced.mp4")
    
    # Abrir el video
    cap = cv2.VideoCapture(ruta_entrada)
    if not cap.isOpened():
        print(f"Error al abrir el video: {ruta_entrada}")
        return
    
    # Obtener propiedades del video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Configurar el escritor de video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(ruta_salida, fourcc, fps, (width, height))
    
    # Crear objeto CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    
    # Kernel para nitidez
    kernel_nitidez = np.array([[0, -1, 0],
                              [-1, 5, -1],
                              [0, -1, 0]])
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Procesando video: {ruta_entrada}")
    print(f"Total frames: {total_frames}")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Mostrar progreso
        frame_count += 1
        if frame_count % 30 == 0:  # Actualizar cada 30 frames
            print(f"Progreso: {frame_count}/{total_frames} frames ({(frame_count/total_frames*100):.1f}%)")
        
        # 1. Aplicar CLAHE
        frame_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(frame_lab)
        l_eq = clahe.apply(l)
        frame_eq = cv2.merge((l_eq, a, b))
        frame_eq = cv2.cvtColor(frame_eq, cv2.COLOR_LAB2BGR)
        
        # 2. Aumentar nitidez
        frame_sharp = cv2.filter2D(frame_eq, -1, kernel_nitidez)
        
        # Guardar frame mejorado
        out.write(frame_sharp)
    
    # Liberar recursos
    cap.release()
    out.release()
    print(f"\nVideo mejorado guardado en: {ruta_salida}")

def procesar_carpeta(carpeta_entrada, carpeta_salida="videos_mejorados"):
    if not os.path.exists(carpeta_entrada):
        print(f"La carpeta de entrada {carpeta_entrada} no existe")
        return
        
    # Procesar cada video MP4 en la carpeta
    for archivo in os.listdir(carpeta_entrada):
        if archivo.lower().endswith('.mp4'):
            ruta_video = os.path.join(carpeta_entrada, archivo)
            mejorar_video(ruta_video, carpeta_salida)

if __name__ == "__main__":
    # Para procesar un solo video:
    # mejorar_video("./dataset/sospechoso/a1.mp4", "videos_mejorados")
    
    # Para procesar todos los videos en una carpeta:
    procesar_carpeta("./dataset/sospechoso", "videos_mejorados")