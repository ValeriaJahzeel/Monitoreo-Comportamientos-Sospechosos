import os
import random
import shutil
from sklearn.model_selection import train_test_split

def split_dataset(images_dir, annotations_dir, output_dir, test_size=0.1, val_size=0.1):
    # Crea las carpetas de salida
    train_images_dir = os.path.join(output_dir, 'train/images')
    train_annotations_dir = os.path.join(output_dir, 'train/labels')
    val_images_dir = os.path.join(output_dir, 'val/images')
    val_annotations_dir = os.path.join(output_dir, 'val/labels')
    test_images_dir = os.path.join(output_dir, 'test/images')
    test_annotations_dir = os.path.join(output_dir, 'test/labels')

    # Crea las carpetas si no existen
    for directory in [train_images_dir, train_annotations_dir, val_images_dir, val_annotations_dir, test_images_dir, test_annotations_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # Obtén la lista de imágenes y archivos de anotaciones
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg') or f.endswith('.png')]
    annotation_files = [f.replace('.jpg', '.txt').replace('.png', '.txt') for f in image_files]

    # Asegúrate de que cada imagen tenga su archivo de anotaciones correspondiente
    assert all(os.path.exists(os.path.join(annotations_dir, ann)) for ann in annotation_files), "Algunas anotaciones no tienen archivo correspondiente."

    # Divide los archivos de imágenes en entrenamiento, validación y prueba
    train_images, temp_images = train_test_split(image_files, test_size=test_size + val_size, random_state=42)
    val_images, test_images = train_test_split(temp_images, test_size=test_size / (test_size + val_size), random_state=42)

    # Función para mover imágenes y archivos de anotaciones
    def move_files(files, source_images_dir, source_annotations_dir, target_images_dir, target_annotations_dir):
        for file in files:
            # Mueve la imagen
            shutil.move(os.path.join(source_images_dir, file), os.path.join(target_images_dir, file))
            # Mueve la anotación
            annotation_file = file.replace('.jpg', '.txt').replace('.png', '.txt')
            shutil.move(os.path.join(source_annotations_dir, annotation_file), os.path.join(target_annotations_dir, annotation_file))

    # Mueve los archivos a las carpetas correspondientes
    move_files(train_images, images_dir, annotations_dir, train_images_dir, train_annotations_dir)
    move_files(val_images, images_dir, annotations_dir, val_images_dir, val_annotations_dir)
    move_files(test_images, images_dir, annotations_dir, test_images_dir, test_annotations_dir)

    print("Dataset dividido en 'train', 'val' y 'test'.")

# Parámetros
images_dir = 'D:\Documentos\Monitoreo-Comportamientos-Sospechosos\people\oxfortd_town_centre\images'  # Ruta a la carpeta con tus imágenes
annotations_dir = 'people/oxfortd_town_centre/labels'  # Ruta a la carpeta con las anotaciones (archivos .txt)
output_dir = 'output'  # Ruta a la carpeta donde se guardarán los conjuntos de entrenamiento, validación y prueba

# Llama a la función para dividir el dataset
split_dataset(images_dir, annotations_dir, output_dir, test_size=0.1, val_size=0.1)
