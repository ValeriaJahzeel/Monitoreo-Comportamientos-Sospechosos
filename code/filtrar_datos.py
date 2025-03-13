import json
import shutil
import os
from multiprocessing import Pool
from tqdm import tqdm

# Rutas del dataset COCO
ANNOTATIONS_PATH = 'D:/Documentos/Monitoreo-Comportamientos-Sospechosos/people/archive/coco2017/annotations/instances_train2017.json'
IMAGES_PATH = 'D:/Documentos/Monitoreo-Comportamientos-Sospechosos/people/archive/coco2017/train2017/'
OUTPUT_IMAGES_PATH = 'filtered_dataset/train/images/'
OUTPUT_LABELS_PATH = 'filtered_dataset/train/labels/'

# Crear carpetas de salida
os.makedirs(OUTPUT_IMAGES_PATH, exist_ok=True)
os.makedirs(OUTPUT_LABELS_PATH, exist_ok=True)

# Cargar las anotaciones del dataset
with open(ANNOTATIONS_PATH) as f:
    data = json.load(f)

# Filtrar las imágenes que contienen solo la clase 'person' (ID=0)
person_images = set()
for ann in data['annotations']:
    if ann['category_id'] == 1:  # COCO usa ID=1 para 'person'
        person_images.add(ann['image_id'])

# Filtrar las imágenes que contienen personas
filtered_images = [img for img in data['images'] if img['id'] in person_images]

# Función para procesar cada imagen y crear las anotaciones en YOLO
def process_image(img):
    img_path = os.path.join(IMAGES_PATH, img['file_name'])
    new_img_path = os.path.join(OUTPUT_IMAGES_PATH, img['file_name'])

    # Copiar la imagen al nuevo directorio
    shutil.copy(img_path, new_img_path)

    # Crear el archivo de anotaciones en formato YOLO
    img_id = img['id']
    label_file = os.path.join(OUTPUT_LABELS_PATH, f"{img['file_name'].split('.')[0]}.txt")

    with open(label_file, 'w') as f:
        for ann in data['annotations']:
            if ann['image_id'] == img_id and ann['category_id'] == 1:
                # Extraer bounding box y convertir a formato YOLO
                x, y, w, h = ann['bbox']
                x_center = (x + w / 2) / img['width']
                y_center = (y + h / 2) / img['height']
                w /= img['width']
                h /= img['height']

                # Escribir en formato YOLO
                f.write(f"0 {x_center} {y_center} {w} {h}\n")

# Usar multiprocessing para procesar las imágenes en paralelo
if __name__ == '__main__':
    # Usamos tqdm para agregar una barra de progreso
    with Pool() as pool:
        # La función map de pool no soporta tqdm directamente, entonces la aplicamos manualmente.
        list(tqdm(pool.imap(process_image, filtered_images), total=len(filtered_images)))

print("✅ Filtrado completo: imágenes y etiquetas guardadas.")
