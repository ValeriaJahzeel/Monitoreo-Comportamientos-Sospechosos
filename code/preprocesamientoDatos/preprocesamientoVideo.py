"""
Script para mejorar la calidad de videos para análisis de movimientos.
Este script aplica varias técnicas de mejora de calidad como:
- Estabilización de video
- Reducción de ruido
- Mejora de contraste
- Mejora de nitidez
- Corrección de iluminación
"""

import cv2
import numpy as np
import os
import argparse
from tqdm import tqdm

class VideoEnhancer:
    def __init__(self, input_path, output_path, 
                 stabilize=True, denoise=True, 
                 enhance_contrast=True, sharpen=True, 
                 correct_lighting=True):
        """
        Inicializa el mejorador de videos
        
        Parámetros:
        -----------
        input_path : str
            Ruta al video de entrada
        output_path : str
            Ruta para guardar el video mejorado
        stabilize : bool
            Aplicar estabilización de video
        denoise : bool
            Aplicar reducción de ruido
        enhance_contrast : bool
            Aplicar mejora de contraste adaptativo
        sharpen : bool
            Aplicar filtro de nitidez
        correct_lighting : bool
            Aplicar corrección de iluminación
        """
        self.input_path = input_path
        self.output_path = output_path
        self.stabilize = stabilize
        self.denoise = denoise
        self.enhance_contrast = enhance_contrast
        self.sharpen = sharpen
        self.correct_lighting = correct_lighting
        
        # Verificar que el archivo de entrada existe
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"No se encontró el archivo: {input_path}")
        
        # Crear directorio de salida si no existe
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def apply_stabilization(self, frames):
        """
        Aplica estabilización de video usando flujo óptico Lucas-Kanade
        """
        print("Aplicando estabilización de video...")
        
        # Parámetros para detección de características
        feature_params = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        )
        
        # Parámetros para Lucas-Kanade
        lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # Convertir primer frame a escala de grises
        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        
        # Encontrar esquinas en el primer frame
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        
        if p0 is None or len(p0) < 10:
            print("No se encontraron suficientes puntos para estabilizar. Omitiendo estabilización.")
            return frames
        
        # Crear una máscara para dibujar
        mask = np.zeros_like(frames[0])
        
        # Matrices de transformación
        transforms = []
        
        # Vector de desplazamiento acumulado
        accumulated_dx = 0
        accumulated_dy = 0
        
        # Ventana de suavizado para matrices de transformación
        smoothing_window = 30
        
        pbar = tqdm(total=len(frames)-1, desc="Estabilizando")
        
        for i in range(1, len(frames)):
            # Convertir a escala de grises
            frame_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            
            # Calcular flujo óptico
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            
            # Si no se encontraron suficientes puntos, usar matriz identidad
            if p1 is None or sum(st) < 10:
                transforms.append(np.eye(2, 3, dtype=np.float32))
                old_gray = frame_gray.copy()
                p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
                if p0 is None:
                    p0 = np.zeros((1, 1, 2), dtype=np.float32)
                pbar.update(1)
                continue
            
            # Seleccionar buenos puntos
            good_new = p1[st == 1]
            good_old = p0[st == 1]
            
            if len(good_new) < 4 or len(good_old) < 4:
                transforms.append(np.eye(2, 3, dtype=np.float32))
                old_gray = frame_gray.copy()
                p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
                if p0 is None:
                    p0 = np.zeros((1, 1, 2), dtype=np.float32)
                pbar.update(1)
                continue
            
            # Estimación de matriz de transformación
            matrix, _ = cv2.estimateAffinePartial2D(good_old, good_new)
            
            if matrix is None:
                transforms.append(np.eye(2, 3, dtype=np.float32))
            else:
                # Aplicar límites a la transformación para evitar movimientos extremos
                dx = matrix[0, 2]
                dy = matrix[1, 2]
                
                # Limitar el desplazamiento a un valor máximo
                max_dx = 30
                max_dy = 30
                dx = max(min(dx, max_dx), -max_dx)
                dy = max(min(dy, max_dy), -max_dy)
                
                # Actualizar la matriz
                matrix[0, 2] = dx
                matrix[1, 2] = dy
                
                transforms.append(matrix)
            
            # Actualizar para el siguiente frame
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)
            
            pbar.update(1)
        
        pbar.close()
        
        # Suavizar trayectoria
        trajectory = np.cumsum(transforms, axis=0)
        smoothed_trajectory = np.copy(trajectory)
        
        # Aplicar filtro promedio para suavizar
        filter_size = min(smoothing_window, len(transforms))
        for i in range(filter_size, len(transforms)):
            smoothed_trajectory[i] = np.mean(trajectory[i-filter_size:i+1], axis=0)
        
        # Calcular trayectoria suave - original para obtener matrices de diferencia
        smoothed_transforms = np.copy(transforms)
        for i in range(len(transforms)):
            if i > 0:
                smoothed_transforms[i] = smoothed_trajectory[i] - smoothed_trajectory[i-1]
            else:
                smoothed_transforms[i] = transforms[i]
        
        # Aplicar transformaciones a los frames
        stabilized_frames = []
        height, width = frames[0].shape[:2]
        
        for i in range(len(frames)):
            if i == 0:
                stabilized_frames.append(frames[i])
                continue
                
            # Aplicar transformación estabilizadora
            frame_stabilized = cv2.warpAffine(
                frames[i], 
                smoothed_transforms[i-1], 
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE
            )
            
            stabilized_frames.append(frame_stabilized)
        
        return stabilized_frames
    
    def apply_denoise(self, frame):
        """
        Aplica reducción de ruido usando filtros
        """
        # Non-local means denoising
        # h: Parámetro de filtrado para componente de luminancia
        # hColor: Parámetro de filtrado para componente de color
        # templateWindowSize: Tamaño de ventana de plantilla
        # searchWindowSize: Tamaño de ventana de búsqueda
        denoised = cv2.fastNlMeansDenoisingColored(
            frame, 
            None, 
            h=10,           # Filtrado de luminancia (valores más altos = más suavizado)
            hColor=10,      # Filtrado de color
            templateWindowSize=7,  # Tamaño de ventana de plantilla
            searchWindowSize=21    # Tamaño de ventana de búsqueda
        )
        return denoised
    
    def apply_contrast_enhancement(self, frame):
        """
        Aplica mejora de contraste adaptativo
        """
        # Convertir a escala de grises para procesar luminancia
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Aplicar ecualización adaptativa de histograma en el canal L
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # Fusionar canales de nuevo
        enhanced_lab = cv2.merge((cl, a, b))
        
        # Convertir de vuelta a BGR
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        return enhanced
    
    def apply_sharpening(self, frame):
        """
        Aplica filtro de nitidez usando el kernel de nitidez
        """
        # Kernel de nitidez
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])
        
        # Aplicar filtro
        sharpened = cv2.filter2D(frame, -1, kernel)
        return sharpened
    
    def apply_lighting_correction(self, frame):
        """
        Aplica corrección de iluminación para normalizar la luz en el video
        """
        # Convertir a escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Aplicar filtro bilateral para suavizar preservando bordes
        smooth = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Calcular la diferencia para obtener sólo detalles
        detail = cv2.subtract(gray, smooth)
        
        # Normalizar la iluminación
        normalized = cv2.normalize(smooth, None, 0, 255, cv2.NORM_MINMAX)
        
        # Añadir detalles de nuevo
        result_gray = cv2.add(normalized, detail)
        
        # Convertir de nuevo a BGR (preparar para fusión)
        result_bgr = cv2.cvtColor(result_gray, cv2.COLOR_GRAY2BGR)
        
        # Preservar color del original
        hsv_original = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, _ = cv2.split(hsv_original)
        
        # Usar valor corregido
        hsv_corrected = cv2.merge([h, s, result_gray])
        
        # Convertir de vuelta a BGR
        corrected = cv2.cvtColor(hsv_corrected, cv2.COLOR_HSV2BGR)
        
        return corrected
    
    def enhance_frame(self, frame):
        """
        Aplica todas las mejoras seleccionadas a un frame
        """
        enhanced = frame.copy()
        
        # Aplicar denoise primero si está habilitado
        if self.denoise:
            enhanced = self.apply_denoise(enhanced)
        
        # Aplicar corrección de iluminación
        if self.correct_lighting:
            enhanced = self.apply_lighting_correction(enhanced)
        
        # Aplicar mejora de contraste
        if self.enhance_contrast:
            enhanced = self.apply_contrast_enhancement(enhanced)
        
        # Aplicar nitidez al final
        if self.sharpen:
            enhanced = self.apply_sharpening(enhanced)
        
        return enhanced
    
    def process(self):
        """
        Procesa el video completo aplicando las mejoras seleccionadas
        """
        # Abrir video de entrada
        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            raise ValueError(f"No se pudo abrir el video: {self.input_path}")
        
        # Obtener propiedades del video
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Procesando video: {self.input_path}")
        print(f"Resolución: {width}x{height}, FPS: {fps}, Frames: {frame_count}")
        
        # Crear objeto VideoWriter para video de salida
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Códec MP4
        out = cv2.VideoWriter(
            self.output_path, 
            fourcc, 
            fps, 
            (width, height)
        )
        
        # Leer todos los frames para estabilización
        frames = []
        if self.stabilize:
            print("Leyendo frames para estabilización...")
            pbar = tqdm(total=frame_count)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
                pbar.update(1)
            pbar.close()
            
            # Aplicar estabilización
            if self.stabilize:
                frames = self.apply_stabilization(frames)
        
            # Procesar frames estabilizados
            print("Aplicando mejoras a frames estabilizados...")
            pbar = tqdm(total=len(frames))
            for frame in frames:
                # Aplicar mejoras adicionales
                enhanced_frame = self.enhance_frame(frame)
                
                # Escribir frame mejorado
                out.write(enhanced_frame)
                pbar.update(1)
            pbar.close()
        else:
            # Si no se aplica estabilización, procesar frame por frame
            print("Aplicando mejoras frame por frame...")
            pbar = tqdm(total=frame_count)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Aplicar mejoras
                enhanced_frame = self.enhance_frame(frame)
                
                # Escribir frame mejorado
                out.write(enhanced_frame)
                pbar.update(1)
            pbar.close()
        
        # Liberar recursos
        cap.release()
        out.release()
        
        print(f"Video mejorado guardado en: {self.output_path}")

def process_multiple_videos(input_dir, output_dir, video_extensions=('.mp4', '.avi', '.mov'), **kwargs):
    """
    Procesa múltiples videos en un directorio
    
    Parámetros:
    -----------
    input_dir : str
        Directorio con videos de entrada
    output_dir : str
        Directorio para guardar videos mejorados
    video_extensions : tuple
        Extensiones de archivos de video a procesar
    **kwargs : dict
        Parámetros adicionales para VideoEnhancer
    """
    # Crear directorio de salida si no existe
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Obtener lista de videos
    videos = []
    for ext in video_extensions:
        videos.extend([f for f in os.listdir(input_dir) if f.lower().endswith(ext)])
    
    if not videos:
        print(f"No se encontraron videos en {input_dir} con extensiones {video_extensions}")
        return
    
    print(f"Se procesarán {len(videos)} videos")
    
    # Procesar cada video
    for i, video_file in enumerate(videos):
        input_path = os.path.join(input_dir, video_file)
        
        # Conservar la estructura de directorios original
        rel_path = os.path.relpath(input_path, input_dir)
        output_path = os.path.join(output_dir, rel_path)
        
        # Asegurar que el directorio de salida existe
        output_file_dir = os.path.dirname(output_path)
        if output_file_dir and not os.path.exists(output_file_dir):
            os.makedirs(output_file_dir)
        
        print(f"\nProcesando video {i+1}/{len(videos)}: {video_file}")
        
        try:
            enhancer = VideoEnhancer(input_path, output_path, **kwargs)
            enhancer.process()
        except Exception as e:
            print(f"Error al procesar {video_file}: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Mejora la calidad de videos para análisis de movimientos.')
    
    # Argumentos para modo de video único
    parser.add_argument('--input', '-i', help='Ruta al video de entrada (modo video único)')
    parser.add_argument('--output', '-o', help='Ruta para guardar el video mejorado (modo video único)')
    
    # Argumentos para modo de procesamiento por lotes
    parser.add_argument('--input-dir', '-id', help='Directorio con videos a procesar (modo por lotes)')
    parser.add_argument('--output-dir', '-od', help='Directorio para guardar videos mejorados (modo por lotes)')
    
    # Opciones de mejora
    parser.add_argument('--no-stabilize', action='store_true', help='Desactivar estabilización de video')
    parser.add_argument('--no-denoise', action='store_true', help='Desactivar reducción de ruido')
    parser.add_argument('--no-contrast', action='store_true', help='Desactivar mejora de contraste')
    parser.add_argument('--no-sharpen', action='store_true', help='Desactivar filtro de nitidez')
    parser.add_argument('--no-lighting', action='store_true', help='Desactivar corrección de iluminación')
    
    args = parser.parse_args()
    
    # Extraer opciones de mejora
    enhance_options = {
        'stabilize': not args.no_stabilize,
        'denoise': not args.no_denoise,
        'enhance_contrast': not args.no_contrast,
        'sharpen': not args.no_sharpen,
        'correct_lighting': not args.no_lighting
    }
    
    # Determinar el modo de operación
    if args.input and args.output:
        # Modo de video único
        enhancer = VideoEnhancer(args.input, args.output, **enhance_options)
        enhancer.process()
    elif args.input_dir and args.output_dir:
        # Modo de procesamiento por lotes
        process_multiple_videos(args.input_dir, args.output_dir, **enhance_options)
    else:
        parser.print_help()
        print("\nError: Debe especificar --input y --output para procesar un solo video, o --input-dir y --output-dir para procesar varios videos.")
        
# python video_enhancer.py --input-dir ./videos_originales --output-dir ./videos_mejorados