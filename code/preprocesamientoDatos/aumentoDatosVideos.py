"""
Script para generar transformaciones en videos para aumentar el conjunto de datos.
Este script complementa al aumentador de CSV generando videos transformados
para el entrenamiento del modelo.
"""
import cv2
import numpy as np
import os
import glob
import random
from pathlib import Path

class VideoAugmenter:
    def __init__(self, input_dir, output_dir):
        """
        Inicializa el aumentador de videos
        
        Parámetros:
        -----------
        input_dir : str
            Directorio que contiene los videos originales
        output_dir : str
            Directorio donde se guardarán los videos aumentados
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        
        # Crear directorio de salida si no existe
        os.makedirs(output_dir, exist_ok=True)
        
        # Obtener videos disponibles
        self.video_files = glob.glob(os.path.join(input_dir, "*.mp4")) + \
                          glob.glob(os.path.join(input_dir, "*.avi"))
        
        print(f"Se encontraron {len(self.video_files)} videos en {input_dir}")
    
    def _aplicar_rotacion(self, frame, angulo):
        """Rota un frame por un ángulo específico"""
        altura, ancho = frame.shape[:2]
        centro = (ancho // 2, altura // 2)
        matriz_rotacion = cv2.getRotationMatrix2D(centro, angulo, 1.0)
        return cv2.warpAffine(frame, matriz_rotacion, (ancho, altura), 
                              borderMode=cv2.BORDER_REPLICATE)
    
    def _aplicar_espejo(self, frame, horizontal=True):
        """Refleja el frame horizontal o verticalmente"""
        if horizontal:
            return cv2.flip(frame, 1)  # 1 = horizontal
        else:
            return cv2.flip(frame, 0)  # 0 = vertical
    
    def _aplicar_zoom(self, frame, factor):
        """Aplica un zoom al frame"""
        altura, ancho = frame.shape[:2]
        centro_x, centro_y = ancho // 2, altura // 2
        
        # Calcular nueva altura y ancho
        nuevo_ancho = int(ancho / factor)
        nuevo_alto = int(altura / factor)
        
        # Calcular coordenadas de recorte
        x1 = max(0, centro_x - nuevo_ancho // 2)
        y1 = max(0, centro_y - nuevo_alto // 2)
        x2 = min(ancho, centro_x + nuevo_ancho // 2)
        y2 = min(altura, centro_y + nuevo_alto // 2)
        
        # Recortar y redimensionar
        recorte = frame[y1:y2, x1:x2]
        if recorte.size == 0:
            return frame
        
        return cv2.resize(recorte, (ancho, altura), interpolation=cv2.INTER_LINEAR)
    
    def _aplicar_cambio_velocidad(self, video_path, output_path, factor):
        """
        Cambia la velocidad del video (factor > 1 acelera, factor < 1 ralentiza)
        Nota: Esto cambia el número de frames del video
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: No se pudo abrir {video_path}")
            return False
        
        # Obtener propiedades del video
        fps_orig = cap.get(cv2.CAP_PROP_FPS)
        fps_nuevo = fps_orig * factor
        ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Inicializar writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output = cv2.VideoWriter(output_path, fourcc, fps_nuevo, (ancho, alto))
        
        # Leer frames y escribir
        if factor >= 1:  # Aceleración
            # Saltar frames
            frame_step = int(factor)
            count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Guardar solo algunos frames
                if count % frame_step == 0:
                    output.write(frame)
                    
                count += 1
        else:  # Ralentización
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Duplicar frames para ralentizar
                for _ in range(int(1/factor)):
                    output.write(frame)
        
        # Liberar recursos
        cap.release()
        output.release()
        
        return True
    
    def _aplicar_cambio_brillo_contraste(self, frame, alpha, beta):
        """
        Ajusta brillo y contraste
        alpha: contraste (1.0 sin cambio)
        beta: brillo (0 sin cambio)
        """
        return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    
    def _aplicar_ruido(self, frame, intensidad=10):
        """Aplica ruido aleatorio al frame"""
        ruido = np.random.normal(0, intensidad, frame.shape).astype(np.uint8)
        return cv2.add(frame, ruido)
    
    def _aplicar_perspectiva(self, frame, intensidad=0.1):
        """Aplica una transformación de perspectiva"""
        altura, ancho = frame.shape[:2]
        
        # Definir puntos de origen
        puntos1 = np.float32([
            [0, 0],
            [ancho, 0],
            [0, altura],
            [ancho, altura]
        ])
        
        # Definir puntos de destino con perturbación
        desplazamiento = int(ancho * intensidad)
        puntos2 = np.float32([
            [random.randint(0, desplazamiento), random.randint(0, desplazamiento)],
            [ancho - random.randint(0, desplazamiento), random.randint(0, desplazamiento)],
            [random.randint(0, desplazamiento), altura - random.randint(0, desplazamiento)],
            [ancho - random.randint(0, desplazamiento), altura - random.randint(0, desplazamiento)]
        ])
        
        # Calcular matriz de transformación y aplicar
        matriz = cv2.getPerspectiveTransform(puntos1, puntos2)
        return cv2.warpPerspective(frame, matriz, (ancho, altura), borderMode=cv2.BORDER_REPLICATE)
    
    def _aplicar_recorte(self, frame, porcentaje=0.8):
        """Recorta una parte del frame y redimensiona al tamaño original"""
        altura, ancho = frame.shape[:2]
        
        # Calcular dimensiones de recorte
        nuevo_ancho = int(ancho * porcentaje)
        nuevo_alto = int(altura * porcentaje)
        
        # Calcular posición del recorte (centrado)
        x = (ancho - nuevo_ancho) // 2
        y = (altura - nuevo_alto) // 2
        
        # Realizar recorte
        recorte = frame[y:y+nuevo_alto, x:x+nuevo_ancho]
        
        # Redimensionar al tamaño original
        return cv2.resize(recorte, (ancho, altura), interpolation=cv2.INTER_LINEAR)
    
    def _aplicar_cambio_color(self, frame, canal, intensidad):
        """Modifica la intensidad de un canal de color específico"""
        # Separar canales
        b, g, r = cv2.split(frame)
        
        # Modificar canal especificado
        if canal == 0:  # Blue
            b = cv2.add(b, intensidad)
        elif canal == 1:  # Green
            g = cv2.add(g, intensidad)
        elif canal == 2:  # Red
            r = cv2.add(r, intensidad)
        
        # Combinar canales
        return cv2.merge([b, g, r])
    
    def aumentar_videos(self, num_aumentados=5, transformaciones=None):
        """
        Genera videos aumentados aplicando diversas transformaciones
        
        Parámetros:
        -----------
        num_aumentados : int
            Número de versiones aumentadas a generar por cada video original
        transformaciones : list
            Lista de transformaciones a aplicar. Si es None, usa todas.
        """
        if not self.video_files:
            print("No se encontraron videos para aumentar")
            return
        
        # Definir todas las transformaciones disponibles
        todas_transformaciones = [
            'rotacion',
            'espejo',
            'zoom',
            'velocidad',
            'brillo_contraste',
            'ruido',
            'perspectiva',
            'recorte',
            'color'
        ]
        
        # Si no se especifican transformaciones, usar todas
        if transformaciones is None:
            transformaciones = todas_transformaciones
        
        print(f"Generando {num_aumentados} versiones aumentadas por video usando: {transformaciones}")
        
        # Procesar cada video
        for video_path in self.video_files:
            nombre_base = Path(video_path).stem
            print(f"\nProcesando video: {nombre_base}")
            
            # Generar versiones aumentadas
            for i in range(num_aumentados):
                # Seleccionar transformaciones aleatorias (2-3)
                num_trans = random.randint(2, min(3, len(transformaciones)))
                trans_seleccionadas = random.sample(transformaciones, num_trans)
                
                # Crear nombre para video aumentado
                trans_str = '_'.join([t[:3] for t in trans_seleccionadas])
                nombre_salida = f"{nombre_base}_aug_{i+1}_{trans_str}.avi"
                ruta_salida = os.path.join(self.output_dir, nombre_salida)
                
                # Verificar si es una transformación de velocidad
                if 'velocidad' in trans_seleccionadas and len(trans_seleccionadas) == 1:
                    # Para cambio de velocidad procesamos el archivo directamente
                    factor = random.uniform(0.5, 2.0)
                    print(f"  Aplicando cambio de velocidad (factor={factor:.2f}) → {nombre_salida}")
                    self._aplicar_cambio_velocidad(video_path, ruta_salida, factor)
                    continue
                
                # Para otras transformaciones, procesamos frame por frame
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print(f"  ✗ Error: No se pudo abrir {video_path}")
                    continue
                
                # Obtener propiedades del video
                fps = cap.get(cv2.CAP_PROP_FPS)
                ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                # Inicializar writer
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                output = cv2.VideoWriter(ruta_salida, fourcc, fps, (ancho, alto))
                
                # Configuración de transformaciones
                config = {}
                if 'rotacion' in trans_seleccionadas:
                    config['rotacion'] = random.uniform(-30, 30)  # Ángulo aleatorio
                if 'espejo' in trans_seleccionadas:
                    config['espejo'] = random.choice([True, False])  # Horizontal o vertical
                if 'zoom' in trans_seleccionadas:
                    config['zoom'] = random.uniform(1.1, 1.5)  # Factor de zoom
                if 'brillo_contraste' in trans_seleccionadas:
                    config['alpha'] = random.uniform(0.7, 1.3)  # Contraste
                    config['beta'] = random.randint(-30, 30)     # Brillo
                if 'ruido' in trans_seleccionadas:
                    config['ruido'] = random.randint(5, 20)  # Intensidad de ruido
                if 'perspectiva' in trans_seleccionadas:
                    config['perspectiva'] = random.uniform(0.05, 0.15)  # Intensidad
                if 'recorte' in trans_seleccionadas:
                    config['recorte'] = random.uniform(0.7, 0.9)  # Porcentaje a mantener
                if 'color' in trans_seleccionadas:
                    config['canal'] = random.randint(0, 2)  # B, G o R
                    config['intensidad'] = random.randint(-40, 40)  # Cambio de intensidad
                
                print(f"  Aplicando transformaciones {trans_seleccionadas} → {nombre_salida}")
                
                # Procesar cada frame
                frame_count = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Aplicar transformaciones seleccionadas
                    for trans in trans_seleccionadas:
                        if trans == 'rotacion':
                            frame = self._aplicar_rotacion(frame, config['rotacion'])
                        elif trans == 'espejo':
                            frame = self._aplicar_espejo(frame, config['espejo'])
                        elif trans == 'zoom':
                            frame = self._aplicar_zoom(frame, config['zoom'])
                        elif trans == 'brillo_contraste':
                            frame = self._aplicar_cambio_brillo_contraste(frame, config['alpha'], config['beta'])
                        elif trans == 'ruido':
                            frame = self._aplicar_ruido(frame, config['ruido'])
                        elif trans == 'perspectiva':
                            frame = self._aplicar_perspectiva(frame, config['perspectiva'])
                        elif trans == 'recorte':
                            frame = self._aplicar_recorte(frame, config['recorte'])
                        elif trans == 'color':
                            frame = self._aplicar_cambio_color(frame, config['canal'], config['intensidad'])
                    
                    # Guardar frame
                    output.write(frame)
                    frame_count += 1
                
                # Liberar recursos
                cap.release()
                output.release()
                
                print(f"  ✓ Generado: {ruta_salida} ({frame_count} frames)")
        
        print("\nProceso de aumento de videos completado.")
    
    def crear_video_composicion(self, num_videos=5):
        """
        Crea videos compuestos mezclando partes de diferentes videos
        
        Parámetros:
        -----------
        num_videos : int
            Número de videos compuestos a generar
        """
        if len(self.video_files) < 2:
            print("Se necesitan al menos 2 videos para crear composiciones")
            return
        
        print(f"Generando {num_videos} videos compuestos...")
        
        for i in range(num_videos):
            # Seleccionar 2-3 videos aleatorios para mezclar
            num_source = random.randint(2, min(3, len(self.video_files)))
            videos_fuente = random.sample(self.video_files, num_source)
            
            # Crear nombre para el video compuesto
            nombres_fuente = [Path(v).stem for v in videos_fuente]
            nombre_comp = f"comp_{i+1}_{'_'.join([n[:3] for n in nombres_fuente])}.avi"
            ruta_salida = os.path.join(self.output_dir, nombre_comp)
            
            print(f"\nCreando video compuesto {i+1}/{num_videos}: {nombre_comp}")
            print(f"  Fuentes: {', '.join(nombres_fuente)}")
            
            # Obtener información de los videos fuente
            info_videos = []
            for v_path in videos_fuente:
                cap = cv2.VideoCapture(v_path)
                if not cap.isOpened():
                    print(f"  ✗ Error: No se pudo abrir {v_path}")
                    continue
                
                info = {
                    'path': v_path,
                    'fps': cap.get(cv2.CAP_PROP_FPS),
                    'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    'frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                }
                
                cap.release()
                info_videos.append(info)
            
            # Calcular propiedades del video compuesto (usar el primer video como base)
            if not info_videos:
                continue
                
            fps = info_videos[0]['fps']
            width = info_videos[0]['width']
            height = info_videos[0]['height']
            
            # Inicializar writer
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            output = cv2.VideoWriter(ruta_salida, fourcc, fps, (width, height))
            
            # Decidir método de composición
            metodo = random.choice(['secuencial', 'intercalado', 'region'])
            print(f"  Método de composición: {metodo}")
            
            if metodo == 'secuencial':
                # Añadir segmentos de cada video en secuencia
                for info in info_videos:
                    # Decidir qué segmento tomar (aleatorio)
                    duracion = min(int(info['frames'] * random.uniform(0.2, 0.4)), 300)  # Máx 300 frames
                    inicio = random.randint(0, max(0, info['frames'] - duracion))
                    
                    cap = cv2.VideoCapture(info['path'])
                    cap.set(cv2.CAP_PROP_POS_FRAMES, inicio)
                    
                    # Copiar frames
                    frame_count = 0
                    while frame_count < duracion:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        # Redimensionar si es necesario
                        if frame.shape[1] != width or frame.shape[0] != height:
                            frame = cv2.resize(frame, (width, height))
                        
                        output.write(frame)
                        frame_count += 1
                    
                    cap.release()
                    print(f"  ✓ Añadidos {frame_count} frames de {Path(info['path']).stem}")
            
            elif metodo == 'intercalado':
                # Intercalar frames de diferentes videos
                caps = []
                for info in info_videos:
                    cap = cv2.VideoCapture(info['path'])
                    caps.append(cap)
                
                # Número total de frames para el video compuesto
                total_frames = random.randint(500, 1000)
                frame_count = 0
                
                while frame_count < total_frames:
                    # Seleccionar un video aleatorio para este frame
                    idx = random.randint(0, len(caps) - 1)
                    
                    ret, frame = caps[idx].read()
                    if not ret:
                        # Reiniciar este video
                        caps[idx].set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = caps[idx].read()
                        if not ret:
                            continue
                    
                    # Redimensionar si es necesario
                    if frame.shape[1] != width or frame.shape[0] != height:
                        frame = cv2.resize(frame, (width, height))
                    
                    output.write(frame)
                    frame_count += 1
                
                # Liberar recursos
                for cap in caps:
                    cap.release()
                
                print(f"  ✓ Generado video intercalado con {frame_count} frames")
            
            elif metodo == 'region':
                # Dividir la pantalla en regiones y mostrar diferentes videos en cada una
                # Cargar todos los videos
                caps = []
                for info in info_videos:
                    cap = cv2.VideoCapture(info['path'])
                    caps.append(cap)
                
                # Decidir disposición: 2x1, 1x2, 2x2
                disposicion = random.choice(['2x1', '1x2', '2x2'])
                
                if disposicion == '2x1':  # Dos videos lado a lado
                    region_width = width // 2
                    region_height = height
                    regiones = [(0, 0), (region_width, 0)]
                elif disposicion == '1x2':  # Dos videos uno encima del otro
                    region_width = width
                    region_height = height // 2
                    regiones = [(0, 0), (0, region_height)]
                else:  # 2x2, cuatro regiones
                    region_width = width // 2
                    region_height = height // 2
                    regiones = [(0, 0), (region_width, 0), (0, region_height), (region_width, region_height)]
                
                # Limitar a los videos disponibles
                num_regiones = min(len(regiones), len(caps))
                regiones = regiones[:num_regiones]
                
                # Generar frames
                total_frames = random.randint(500, 1000)
                frame_count = 0
                
                while frame_count < total_frames:
                    # Inicializar frame compuesto
                    frame_comp = np.zeros((height, width, 3), dtype=np.uint8)
                    
                    # Añadir cada región
                    for i, (x, y) in enumerate(regiones):
                        ret, frame = caps[i].read()
                        if not ret:
                            # Reiniciar este video
                            caps[i].set(cv2.CAP_PROP_POS_FRAMES, 0)
                            ret, frame = caps[i].read()
                            if not ret:
                                continue
                        
                        # Redimensionar al tamaño de la región
                        frame_resized = cv2.resize(frame, (region_width, region_height))
                        
                        # Añadir al frame compuesto
                        frame_comp[y:y+region_height, x:x+region_width] = frame_resized
                    
                    output.write(frame_comp)
                    frame_count += 1
                
                # Liberar recursos
                for cap in caps:
                    cap.release()
                
                print(f"  ✓ Generado video multi-región ({disposicion}) con {frame_count} frames")