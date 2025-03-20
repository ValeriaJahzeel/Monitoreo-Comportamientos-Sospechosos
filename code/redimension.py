import cv2
import os

# Rutas
input_path = r'D:\Documentos\Monitoreo-Comportamientos-Sospechosos\dataset\normal'
output_path = r'D:\Documentos\Monitoreo-Comportamientos-Sospechosos\dataset\normal_reducido'

# Crear el directorio de salida si no existe
os.makedirs(output_path, exist_ok=True)

# Dimensiones deseadas
width = 640
height = 360

with os.scandir(input_path) as ficheros:
    for fichero in ficheros:
        if fichero.name.endswith(('.mp4', '.avi', '.mov')):
            print(f"Procesando: {fichero.name}")
            
            # Ruta completa del video
            input_file = os.path.join(input_path, fichero.name)
            output_file = os.path.join(output_path, f"{fichero.name}")

            try:
                cap = cv2.VideoCapture(input_file)

                # Verifica si el video se abrió correctamente
                if not cap.isOpened():
                    print(f"No se pudo abrir el archivo: {fichero.name}")
                    continue

                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_file, fourcc, cap.get(cv2.CAP_PROP_FPS), (width, height))

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Redimensionar el frame
                    resized_frame = cv2.resize(frame, (width, height))
                    out.write(resized_frame)

                cap.release()
                out.release()
                print(f"Video redimensionado correctamente: {fichero.name}")

            except Exception as e:
                print(f"Error al procesar {fichero.name}: {e}")
