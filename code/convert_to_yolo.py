import cv2
import os

def resize_coordinates(x, y, width, height, orig_width, orig_height, new_width, new_height):
    # Calcula las coordenadas normalizadas según la nueva resolución
    x_new = (x * new_width) / orig_width
    y_new = (y * new_height) / orig_height
    width_new = (width * new_width) / orig_width
    height_new = (height * new_height) / orig_height
    return x_new, y_new, width_new, height_new

def extract_frames_and_convert_to_yolo(input_video, input_data, output_folder, image_width, image_height, new_width, new_height):
    # Asegúrate de que la carpeta de salida exista
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Crea carpetas para las imágenes y las anotaciones
    images_folder = os.path.join(output_folder, "images")
    annotations_folder = os.path.join(output_folder, "annotations")
    
    if not os.path.exists(images_folder):
        os.makedirs(images_folder)
    
    if not os.path.exists(annotations_folder):
        os.makedirs(annotations_folder)

    # Abre el video
    cap = cv2.VideoCapture(input_video)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    # Lee las líneas del archivo de datos
    with open(input_data, 'r') as infile:
        lines = infile.readlines()

    # Crea un diccionario para almacenar las anotaciones por frame
    frame_annotations = {}

    # Itera sobre las líneas del archivo de datos
    for line in lines:
        line = line.strip()
        data = line.split(',')

        # Verifica si la línea tiene la cantidad correcta de datos (12 valores)
        if len(data) != 12:
            continue

        frame_number = int(data[1])
        head_left = float(data[4])
        head_top = float(data[5])
        head_right = float(data[6])
        head_bottom = float(data[7])

        body_left = float(data[8])
        body_top = float(data[9])
        body_right = float(data[10])
        body_bottom = float(data[11])

        # Calcula las coordenadas normalizadas para la cabeza en resolución original (video)
        head_x_center = (head_left + head_right) / 2 / image_width
        head_y_center = (head_top + head_bottom) / 2 / image_height
        head_width = (head_right - head_left) / image_width
        head_height = (head_bottom - head_top) / image_height

        # Calcula las coordenadas normalizadas para el cuerpo en resolución original (video)
        body_x_center = (body_left + body_right) / 2 / image_width
        body_y_center = (body_top + body_bottom) / 2 / image_height
        body_width = (body_right - body_left) / image_width
        body_height = (body_bottom - body_top) / image_height

        # Si el frame no existe en el diccionario, lo inicializamos
        if frame_number not in frame_annotations:
            frame_annotations[frame_number] = []

        # Ajusta las coordenadas normalizadas a la nueva resolución (640x640)
        head_x_center, head_y_center, head_width, head_height = resize_coordinates(
            head_x_center, head_y_center, head_width, head_height, image_width, image_height, new_width, new_height)
        body_x_center, body_y_center, body_width, body_height = resize_coordinates(
            body_x_center, body_y_center, body_width, body_height, image_width, image_height, new_width, new_height)

        # Agrega las anotaciones para cabeza y cuerpo al diccionario
        frame_annotations[frame_number].append(f"0 {head_x_center} {head_y_center} {head_width} {head_height}")
        frame_annotations[frame_number].append(f"0 {body_x_center} {body_y_center} {body_width} {body_height}")

    # Extrae los fotogramas y guarda las anotaciones
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Fin del video

        # Si el número de frame está en el diccionario, guarda la imagen y las anotaciones
        if frame_count in frame_annotations:
            # Guarda el fotograma como imagen
            frame_filename = os.path.join(images_folder, f"frame_{frame_count:04d}.jpg")
            frame_resized = cv2.resize(frame, (new_width, new_height))  # Redimensiona el fotograma
            cv2.imwrite(frame_filename, frame_resized)

            # Guarda las anotaciones en formato YOLO
            annotations_filename = os.path.join(annotations_folder, f"frame_{frame_count:04d}.txt")
            with open(annotations_filename, 'w') as outfile:
                for annotation in frame_annotations[frame_count]:
                    outfile.write(annotation + "\n")

        frame_count += 1

    # Cierra el video
    cap.release()

# Parámetros
input_video = 'D:\Documentos\Monitoreo-Comportamientos-Sospechosos\people\oxfortd_town_centre\TownCentreXVID.mp4'  # Ruta al archivo de video
input_data = 'D:\Documentos\Monitoreo-Comportamientos-Sospechosos\people\oxfortd_town_centre\data.txt'  # Ruta al archivo con los datos
output_folder = 'annotations'  # Carpeta donde se guardarán los archivos de anotaciones e imágenes
image_width = 1920  # Ancho de las imágenes del video
image_height = 1080  # Alto de las imágenes del video
new_width = 640  # Nuevo ancho para las imágenes de entrenamiento
new_height = 640  # Nuevo alto para las imágenes de entrenamiento

# Llama a la función
extract_frames_and_convert_to_yolo(input_video, input_data, output_folder, image_width, image_height, new_width, new_height)
