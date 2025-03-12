import os
import cv2

# Directorio de entrada
input_dir = "dataset"  # Carpeta original con las imágenes
output_dir = "normalized_dataset"  # Carpeta donde se guardarán las imágenes normalizadas

# Dimensiones deseadas
target_size = (96, 96)

# Crear la carpeta de salida si no existe
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Procesar cada carpeta/clase
for class_name in os.listdir(input_dir):
    class_path = os.path.join(input_dir, class_name)
    
    # Verificar si es un directorio
    if not os.path.isdir(class_path):
        continue
    
    # Crear subcarpeta en la salida
    output_class_path = os.path.join(output_dir, class_name)
    if not os.path.exists(output_class_path):
        os.makedirs(output_class_path)
    
    # Procesar cada imagen en la carpeta
    for i, filename in enumerate(os.listdir(class_path)):
        input_image_path = os.path.join(class_path, filename)
        
        # Leer la imagen
        try:
            img = cv2.imread(input_image_path)
            if img is None:
                print(f"Error al leer la imagen: {input_image_path}")
                continue
            
            # Cambiar el tamaño
            img_resized = cv2.resize(img, target_size)
            
            # Renombrar y guardar la imagen
            output_image_name = f"{class_name}_{i+1:04d}.jpg"
            output_image_path = os.path.join(output_class_path, output_image_name)
            cv2.imwrite(output_image_path, img_resized)
            print(f"Guardada: {output_image_path}")
        
        except Exception as e:
            print(f"Error procesando {input_image_path}: {e}")