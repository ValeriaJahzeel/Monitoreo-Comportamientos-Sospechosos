import cv2
import os
import argparse
from tqdm import tqdm

def flip_video(input_file, output_file):
    """
    Voltea un video horizontalmente (efecto espejo)
    
    Args:
        input_file (str): Ruta al archivo de video de entrada
        output_file (str): Ruta al archivo de video de salida
    """
    # Abrir el archivo de video
    cap = cv2.VideoCapture(input_file)
    
    # Verificar si el video se abrió correctamente
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el video {input_file}")
        return False
    
    # Obtener propiedades del video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Definir el codec y crear el objeto VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Puede cambiar según el formato de salida deseado
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    
    # Procesar cada frame
    progress_bar = tqdm(total=total_frames, desc=f"Procesando {os.path.basename(input_file)}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Voltear el frame horizontalmente (efecto espejo)
        flipped_frame = cv2.flip(frame, 1)
        
        # Escribir el frame en el archivo de salida
        out.write(flipped_frame)
        
        # Actualizar la barra de progreso
        progress_bar.update(1)
    
    # Cerrar los objetos
    progress_bar.close()
    cap.release()
    out.release()
    
    print(f"Video procesado: {output_file}")
    return True

def process_directory(input_dir, output_dir, extensions=['.mp4', '.avi', '.mov', '.mkv']):
    """
    Procesa todos los videos en un directorio
    
    Args:
        input_dir (str): Directorio con los videos originales
        output_dir (str): Directorio donde se guardarán los videos procesados
        extensions (list): Lista de extensiones de archivo a procesar
    """
    # Crear el directorio de salida si no existe
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Contar los archivos de video en el directorio
    video_files = []
    for file in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file)
        if os.path.isfile(file_path) and any(file.lower().endswith(ext) for ext in extensions):
            video_files.append(file)
    
    print(f"Encontrados {len(video_files)} videos para procesar.")
    
    # Encontrar el número más alto en los archivos existentes en el directorio de salida
    highest_number = 27
    for file in os.listdir(output_dir):
        if os.path.isfile(os.path.join(output_dir, file)) and any(file.lower().endswith(ext) for ext in extensions):
            try:
                # Intentar extraer un número del nombre del archivo
                file_name = os.path.splitext(file)[0]
                if file_name.isdigit():
                    file_number = int(file_name)
                    highest_number = max(highest_number, file_number)
            except (ValueError, TypeError):
                continue
    
    print(f"El número más alto encontrado en archivos existentes es: {highest_number}")
    
    # Procesar cada video
    for i, video_file in enumerate(video_files):
        input_path = os.path.join(input_dir, video_file)
        # El nuevo nombre será el siguiente número después del más alto encontrado
        new_number = highest_number + i + 1
        # Obtener la extensión del archivo original
        extension = os.path.splitext(video_file)[1]
        output_path = os.path.join(output_dir, f"{new_number}{extension}")
        flip_video(input_path, output_path)
    
    print(f"Proceso completado. {len(video_files)} videos han sido reflejados (efecto espejo) y guardados en {output_dir}")

if __name__ == "__main__":
    # Configurar el parser de argumentos
    parser = argparse.ArgumentParser(description="Voltear videos horizontalmente (efecto espejo)")
    parser.add_argument("-i", "--input", help="Directorio de entrada con videos", required=True)
    parser.add_argument("-o", "--output", help="Directorio de salida para videos procesados", required=True)
    args = parser.parse_args()
    
    # Procesar el directorio
    process_directory(args.input, args.output)