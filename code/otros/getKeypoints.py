import cv2
import numpy as np
import os
import csv
import shutil
import mediapipe as mp
import code.otros.getFrame as gf

""" 
Esta sección extrae los frames de un video y guarda las 
coordenadas de los landmarks en un archivo CSV 

Ocupa MEDIAPIPE para obtener los landmarks de las poses
"""

def framesVideos(rutaVideos, rutaFrames, rutaCSV, rutaTrazos):
    for i in range(1, 4):
        video = os.path.join(rutaVideos, f"{i}.mp4")
        frames = os.path.join(rutaFrames, str(i))
        csv_file = os.path.join(rutaCSV, f"{i}.csv")
        trazos = os.path.join(rutaTrazos, str(i))

        # Eliminar y recrear carpetas
        for carpeta in [frames, trazos]:
            if os.path.exists(carpeta):
                shutil.rmtree(carpeta)
                print(f"Se eliminó la carpeta: {carpeta}")
            os.makedirs(carpeta)
            print(f"Carpeta creada: {carpeta}")

        gf.ObtenerFrames(video, frames, trazos)

        # Obtener la lista de frames del video
        nombres = sorted([os.path.join(frames, f) for f in os.listdir(frames)])

        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_pose = mp.solutions.pose

        # Color de fondo para la segmentación
        BG_COLOR = (192, 192, 192)  # Gris

        # Crear archivo CSV
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['image', 'landmark', 'x', 'y', 'z', 'visibility'])

            with mp_pose.Pose(
                static_image_mode=True,
                model_complexity=2,
                enable_segmentation=True,
                min_detection_confidence=0.5
            ) as pose:

                for idx, file in enumerate(nombres):
                    image = cv2.imread(file)
                    if image is None:
                        print(f"No se pudo leer la imagen: {file}")
                        continue

                    image_height, image_width, _ = image.shape
                    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                    if not results.pose_landmarks:
                        continue

                    # Guardar coordenadas de los landmarks en el CSV
                    for landmark_id, landmark in enumerate(results.pose_landmarks.landmark):
                        writer.writerow([
                            f'frame{idx}.png',
                            mp_pose.PoseLandmark(landmark_id).name,
                            landmark.x * image_width,
                            landmark.y * image_height,
                            landmark.z,
                            landmark.visibility
                        ])

                    # Dibujar los landmarks en la imagen
                    annotated_image = image.copy()
                    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
                    bg_image = np.zeros(image.shape, dtype=np.uint8)
                    bg_image[:] = BG_COLOR
                    annotated_image = np.where(condition, annotated_image, bg_image)

                    mp_drawing.draw_landmarks(
                        annotated_image,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                    )

                    # Guardar la imagen anotada
                    ruta_guardado = os.path.join(trazos, f'frame{idx}.png')
                    cv2.imwrite(ruta_guardado, annotated_image)
                    print(f"Guardado: {ruta_guardado}")
