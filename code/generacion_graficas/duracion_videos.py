import cv2
import os
import matplotlib.pyplot as plt
import pandas as pd  # Faltaba este import

def obtener_duracion_video(ruta_video):
    cap = cv2.VideoCapture(ruta_video)
    if not cap.isOpened():
        print(f"⚠️ No se pudo abrir el video: {ruta_video}")
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duracion_seg = total_frames / fps if fps else 0
    cap.release()
    return duracion_seg

def graficar_duracion_videos_varias_carpetas(directorios):
    nombres_videos = []
    duraciones = []
    carpetas_origen = []

    for carpeta in directorios:
        if not os.path.exists(carpeta):
            print(f"⚠️ Carpeta no encontrada: {carpeta}")
            continue

        nombre_carpeta = os.path.basename(carpeta)
        for archivo in os.listdir(carpeta):
            if archivo.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                ruta = os.path.join(carpeta, archivo)
                duracion = obtener_duracion_video(ruta)
                if duracion is not None:
                    nombres_videos.append(archivo)
                    duraciones.append(duracion)
                    carpetas_origen.append(nombre_carpeta)

    if not duraciones:
        print("❌ No se encontraron videos válidos.")
        return

    # Crear DataFrame
    df = pd.DataFrame({
        "Clase": carpetas_origen,
        "Duracion": duraciones
    })

    # Gráfica de boxplot por clase
    plt.figure(figsize=(10, 6))
    df.boxplot(column="Duracion", by="Clase", grid=False)
    plt.title("Distribución de duración por clase")
    plt.suptitle("")  # Eliminar título duplicado que agrega pandas
    plt.ylabel("Duración (segundos)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Directorios con tus clases de videos
carpetas = [
    r"D:\Documentos\Monitoreo-Comportamientos-Sospechosos\dataset\normal",
    r"D:\Documentos\Monitoreo-Comportamientos-Sospechosos\dataset\sospechoso\forcejeo",
    r"D:\Documentos\Monitoreo-Comportamientos-Sospechosos\dataset\sospechoso\merodeo"
]

graficar_duracion_videos_varias_carpetas(carpetas)
