"""
Módulo para técnicas de aumento de datos en análisis de movimiento y comportamiento.
Este script proporciona funciones para incrementar el volumen de datos disponibles
a partir de un conjunto limitado de archivos CSV de trayectorias.
"""
import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
import glob
import random
import math
import warnings
warnings.filterwarnings('ignore')

class MovementDataAugmenter:
    def __init__(self, csv_directory, output_directory):
        """
        Inicializa el aumentador de datos de movimiento
        
        Parámetros:
        -----------
        csv_directory : str
            Ruta al directorio que contiene los archivos CSV originales
        output_directory : str
            Ruta donde se guardarán los archivos CSV aumentados
        """
        self.csv_directory = csv_directory
        self.output_directory = output_directory
        self.dataframes = {}  # Almacenar DataFrames originales
        self.metadata = {}    # Almacenar metadatos (etiquetas, patrones, etc)
        
        # Crear directorio de salida si no existe
        os.makedirs(output_directory, exist_ok=True)
    
    def cargar_datos(self):
        """Carga todos los archivos CSV del directorio especificado"""
        csv_files = glob.glob(os.path.join(self.csv_directory, "*.csv"))
        
        if not csv_files:
            print(f"No se encontraron archivos CSV en {self.csv_directory}")
            return False
        
        print(f"Cargando {len(csv_files)} archivos CSV originales...")
        
        for csv_file in csv_files:
            nombre_video = os.path.basename(csv_file).split('.')[0]
            try:
                df = pd.read_csv(csv_file)
                self.dataframes[nombre_video] = df
                
                # Extraer metadatos básicos para cada archivo
                self.metadata[nombre_video] = {
                    'duracion_frames': df['Frame'].max() - df['Frame'].min() + 1,
                    'objetos': df['Objeto'].nunique(),
                    'patrones': df['Patron_Movimiento'].value_counts().to_dict() if 'Patron_Movimiento' in df.columns else {}
                }
                
                print(f"  ✓ Cargado {nombre_video}: {len(df)} registros")
            except Exception as e:
                print(f"  ✗ Error al cargar {nombre_video}: {str(e)}")
        
        print(f"Total de archivos cargados: {len(self.dataframes)}")
        return len(self.dataframes) > 0
    
    def _interpolar_trayectoria(self, df, factor=2.0):
        """
        Interpola puntos en la trayectoria para crear movimientos más suaves y duplicar datos
        
        Parámetros:
        -----------
        df : DataFrame
            DataFrame con datos de trayectoria
        factor : float
            Factor de interpolación (2.0 = duplicar puntos)
        """
        # Agrupar por objeto para procesar cada trayectoria por separado
        objetos = df['Objeto'].unique()
        frames = np.sort(df['Frame'].unique())
        
        # Nuevo DataFrame para almacenar resultados
        nuevos_registros = []
        
        for obj in objetos:
            # Extraer trayectoria de este objeto
            traj = df[df['Objeto'] == obj].sort_values('Frame')
            
            if len(traj) < 3:  # Necesitamos al menos 3 puntos para interpolar
                nuevos_registros.append(traj)
                continue
            
            # Crear nuevos frames interpolados
            frames_orig = traj['Frame'].values
            n_frames = len(frames_orig)
            frames_nuevos = np.linspace(frames_orig.min(), frames_orig.max(), 
                                        int(n_frames * factor))
            
            # Interpolar columnas numéricas
            columnas_numericas = traj.select_dtypes(include=['float64', 'int64']).columns
            columnas_numericas = [c for c in columnas_numericas if c != 'Frame' and c != 'Objeto']
            
            # Crear interpoladores para cada columna numérica
            interpoladores = {}
            for col in columnas_numericas:
                # Usar interpolación cúbica cuando hay suficientes puntos, lineal si no
                kind = 'cubic' if len(traj) >= 4 else 'linear'
                interpoladores[col] = interp1d(frames_orig, traj[col].values, 
                                              kind=kind, bounds_error=False, 
                                              fill_value='extrapolate')
            
            # Crear nuevos registros interpolados
            for frame_nuevo in frames_nuevos:
                nuevo_registro = {'Frame': frame_nuevo, 'Objeto': obj}
                
                for col in columnas_numericas:
                    nuevo_registro[col] = float(interpoladores[col](frame_nuevo))
                
                # Copiar columnas categóricas del frame más cercano
                frame_cercano = frames_orig[np.argmin(np.abs(frames_orig - frame_nuevo))]
                registro_cercano = traj[traj['Frame'] == frame_cercano].iloc[0]
                
                for col in traj.select_dtypes(include=['object']).columns:
                    nuevo_registro[col] = registro_cercano[col]
                
                nuevos_registros.append(nuevo_registro)
        
        return pd.DataFrame(nuevos_registros)
    
    def _perturbar_velocidad(self, df, factor_min=0.8, factor_max=1.2):
        """
        Perturba la velocidad y aceleración manteniendo la trayectoria
        
        Parámetros:
        -----------
        df : DataFrame
            DataFrame con datos de trayectoria
        factor_min, factor_max : float
            Rango de factores para perturbar la velocidad
        """
        # Crear copia para no modificar el original
        df_perturbado = df.copy()
        
        # Columnas a perturbar
        columnas_velocidad = ['Velocidad', 'Aceleracion', 'Desplazamiento']
        
        # Perturbar por objeto para mantener consistencia
        for obj in df['Objeto'].unique():
            # Generar factor de perturbación para este objeto
            factor = np.random.uniform(factor_min, factor_max)
            
            # Aplicar perturbación
            for col in columnas_velocidad:
                if col in df.columns:
                    mask = df_perturbado['Objeto'] == obj
                    df_perturbado.loc[mask, col] = df_perturbado.loc[mask, col] * factor
        
        return df_perturbado
    
    def _invertir_direccion(self, df):
        """
        Invierte la dirección del movimiento manteniendo el patrón
        """
        # Crear copia para no modificar el original
        df_invertido = df.copy()
        
        # Si existe la columna Dirección, invertir (añadir 180° módulo 360)
        if 'Direccion' in df.columns:
            df_invertido['Direccion'] = (df_invertido['Direccion'] + 180) % 360
        
        return df_invertido
    
    def _rotar_trayectoria(self, df, angulo_grados):
        """
        Rota la trayectoria un ángulo específico
        
        Parámetros:
        -----------
        df : DataFrame
            DataFrame con datos de trayectoria
        angulo_grados : float
            Ángulo de rotación en grados
        """
        # Crear copia para no modificar el original
        df_rotado = df.copy()
        
        # Convertir ángulo a radianes
        angulo_rad = np.radians(angulo_grados)
        
        # Matriz de rotación 2D
        cos_a = np.cos(angulo_rad)
        sin_a = np.sin(angulo_rad)
        
        # Rotar por objeto
        for obj in df['Objeto'].unique():
            mask = df_rotado['Objeto'] == obj
            
            # Asegurar que tenemos las coordenadas
            if 'Centroide_X' in df.columns and 'Centroide_Y' in df.columns:
                # Extraer coordenadas
                x = df_rotado.loc[mask, 'Centroide_X'].values
                y = df_rotado.loc[mask, 'Centroide_Y'].values
                
                # Calcular centro de la trayectoria
                cx = np.mean(x)
                cy = np.mean(y)
                
                # Trasladar al origen
                x_origen = x - cx
                y_origen = y - cy
                
                # Aplicar rotación
                x_rotado = x_origen * cos_a - y_origen * sin_a
                y_rotado = x_origen * sin_a + y_origen * cos_a
                
                # Trasladar de vuelta
                df_rotado.loc[mask, 'Centroide_X'] = x_rotado + cx
                df_rotado.loc[mask, 'Centroide_Y'] = y_rotado + cy
                
                # Actualizar dirección si existe
                if 'Direccion' in df.columns:
                    df_rotado.loc[mask, 'Direccion'] = (df_rotado.loc[mask, 'Direccion'] + angulo_grados) % 360
        
        return df_rotado
    
    def _generar_variante_aumento_objetos(self, df, factor=1.5):
        """
        Genera una variante con más objetos combinando trayectorias aleatorias del dataset
        
        Parámetros:
        -----------
        df : DataFrame
            DataFrame con datos de trayectoria
        factor : float
            Factor de aumento de objetos (1.5 = 50% más objetos)
        """
        df_aumentado = df.copy()
        
        # Obtener objetos originales
        objetos_orig = df['Objeto'].unique()
        num_objetos = len(objetos_orig)
        objetos_a_agregar = int(num_objetos * (factor - 1)) + 1
        
        # Crear nuevos IDs para los objetos a agregar
        ultimo_id = max([int(str(obj).replace('obj_', '')) 
                        if str(obj).startswith('obj_') else 0 
                        for obj in objetos_orig]) + 1
        
        # Agregar nuevos objetos
        for i in range(objetos_a_agregar):
            # Seleccionar un objeto aleatorio para clonar
            obj_orig = np.random.choice(objetos_orig)
            nuevo_obj_id = f"obj_{ultimo_id + i}"
            
            # Clonar datos
            datos_obj = df[df['Objeto'] == obj_orig].copy()
            datos_obj['Objeto'] = nuevo_obj_id
            
            # Modificar posición para evitar superposición exacta
            offset_x = np.random.uniform(-50, 50)
            offset_y = np.random.uniform(-50, 50)
            
            if 'Centroide_X' in datos_obj.columns and 'Centroide_Y' in datos_obj.columns:
                datos_obj['Centroide_X'] += offset_x
                datos_obj['Centroide_Y'] += offset_y
            
            # Perturbar velocidad y otras métricas
            datos_obj = self._perturbar_velocidad(datos_obj)
            
            # Añadir al dataframe
            df_aumentado = pd.concat([df_aumentado, datos_obj], ignore_index=True)
        
        return df_aumentado
    
    def _mezclar_videos(self, df1, df2, proporcion=0.5):
        """
        Crea un nuevo video mezclando objetos de dos videos diferentes
        
        Parámetros:
        -----------
        df1, df2 : DataFrame
            DataFrames de dos videos a mezclar
        proporcion : float
            Proporción de objetos a tomar del segundo video (0.5 = 50%)
        """
        # Crear copia del primer video
        df_mezclado = df1.copy()
        
        # Extraer objetos del segundo video
        objetos_df2 = df2['Objeto'].unique()
        num_objetos_a_tomar = max(1, int(len(objetos_df2) * proporcion))
        objetos_seleccionados = np.random.choice(objetos_df2, 
                                                size=num_objetos_a_tomar, 
                                                replace=False)
        
        # Crear nuevos IDs para evitar colisiones
        objetos_df1 = df1['Objeto'].unique()
        ultimo_id = max([int(str(obj).replace('obj_', '')) 
                        if str(obj).startswith('obj_') else 0 
                        for obj in objetos_df1]) + 1
        
        # Añadir objetos seleccionados con nuevos IDs
        for i, obj in enumerate(objetos_seleccionados):
            datos_obj = df2[df2['Objeto'] == obj].copy()
            
            # Ajustar IDs
            nuevo_obj_id = f"obj_{ultimo_id + i}"
            datos_obj['Objeto'] = nuevo_obj_id
            
            # Ajustar posiciones si es necesario
            if 'Centroide_X' in datos_obj.columns and 'Centroide_Y' in datos_obj.columns:
                # Calcular centroides medios de ambos dataframes
                centro1_x = df1['Centroide_X'].mean() if 'Centroide_X' in df1.columns else 0
                centro1_y = df1['Centroide_Y'].mean() if 'Centroide_Y' in df1.columns else 0
                
                centro2_x = df2['Centroide_X'].mean() if 'Centroide_X' in df2.columns else 0
                centro2_y = df2['Centroide_Y'].mean() if 'Centroide_Y' in df2.columns else 0
                
                # Calcular offset
                offset_x = centro1_x - centro2_x
                offset_y = centro1_y - centro2_y
                
                # Aplicar ajuste y añadir desplazamiento aleatorio
                datos_obj['Centroide_X'] += offset_x + np.random.uniform(-30, 30)
                datos_obj['Centroide_Y'] += offset_y + np.random.uniform(-30, 30)
            
            # Añadir al dataframe mezclado
            df_mezclado = pd.concat([df_mezclado, datos_obj], ignore_index=True)
        
        return df_mezclado
    
    def _aplicar_ruido_gaussiano(self, df, nivel_ruido=0.05):
        """
        Aplica ruido gaussiano a variables numéricas para simular variaciones
        
        Parámetros:
        -----------
        df : DataFrame
            DataFrame con datos de trayectoria
        nivel_ruido : float
            Intensidad del ruido (desviación estándar como proporción de la media)
        """
        # Crear copia para no modificar el original
        df_ruidoso = df.copy()
        
        # Identificar columnas numéricas
        columnas_num = df.select_dtypes(include=['float64', 'int64']).columns
        columnas_num = [c for c in columnas_num if c != 'Frame' and c != 'Objeto']
        
        # Aplicar ruido por columna
        for col in columnas_num:
            # Calcular nivel de ruido específico para esta columna
            media = np.abs(df[col].mean())
            sigma = media * nivel_ruido if media > 0 else 0.1
            
            # Generar ruido
            ruido = np.random.normal(0, sigma, size=len(df_ruidoso))
            
            # Aplicar ruido
            df_ruidoso[col] = df_ruidoso[col] + ruido
        
        return df_ruidoso
    
    def _generar_fragmento(self, df, inicio_rel=0.0, fin_rel=1.0):
        """
        Genera un fragmento de video basado en proporciones relativas del original
        
        Parámetros:
        -----------
        df : DataFrame
            DataFrame con datos de trayectoria
        inicio_rel, fin_rel : float
            Proporciones relativas de inicio y fin (0.0 a 1.0)
        """
        # Obtener rango de frames
        frame_min = df['Frame'].min()
        frame_max = df['Frame'].max()
        rango_frames = frame_max - frame_min
        
        # Calcular frames de inicio y fin
        frame_inicio = int(frame_min + rango_frames * inicio_rel)
        frame_fin = int(frame_min + rango_frames * fin_rel)
        
        # Extraer fragmento
        fragmento = df[(df['Frame'] >= frame_inicio) & (df['Frame'] <= frame_fin)].copy()
        
        # Reindexar frames para que comiencen desde 0
        fragmento['Frame'] = fragmento['Frame'] - frame_inicio
        
        return fragmento
    
    def generar_dataset_aumentado(self, num_aumentados=100, estrategias=None):
        """
        Genera un dataset aumentado aplicando múltiples estrategias
        
        Parámetros:
        -----------
        num_aumentados : int
            Número de archivos aumentados a generar
        estrategias : list
            Lista de estrategias a aplicar. Si es None, usa todas.
        """
        if not self.dataframes:
            if not self.cargar_datos():
                print("No se pudieron cargar los datos originales")
                return False
        
        # Listar todas las estrategias disponibles
        todas_estrategias = [
            'interpolar',           # Interpolar puntos intermedios
            'perturbar_velocidad',  # Cambiar velocidad
            'invertir_direccion',   # Invertir dirección
            'rotar',                # Rotar trayectoria
            'aumentar_objetos',     # Añadir más objetos
            'mezclar_videos',       # Mezclar objetos de diferentes videos
            'ruido_gaussiano',      # Añadir ruido aleatorio
            'fragmentar'            # Crear fragmentos de videos
        ]
        
        # Si no se especifican estrategias, usar todas
        if estrategias is None:
            estrategias = todas_estrategias
        
        print(f"Generando {num_aumentados} archivos aumentados usando estrategias: {estrategias}")
        
        # Obtener lista de archivos originales
        nombres_originales = list(self.dataframes.keys())
        
        # Contadores para seguimiento
        archivos_generados = 0
        
        # Generar archivos aumentados
        for i in range(num_aumentados):
            # Seleccionar archivo base aleatorio
            nombre_base = np.random.choice(nombres_originales)
            df_base = self.dataframes[nombre_base].copy()
            
            # Inicializar DataFrame aumentado
            df_aumentado = df_base.copy()
            
            # Seleccionar estrategias aleatorias a aplicar (2-4 estrategias)
            num_estrategias = np.random.randint(2, min(5, len(estrategias) + 1))
            estrategias_aplicar = np.random.choice(estrategias, 
                                                  size=num_estrategias, 
                                                  replace=False)
            
            # Aplicar cada estrategia seleccionada
            for estrategia in estrategias_aplicar:
                if estrategia == 'interpolar':
                    factor = np.random.uniform(1.5, 3.0)
                    df_aumentado = self._interpolar_trayectoria(df_aumentado, factor)
                
                elif estrategia == 'perturbar_velocidad':
                    factor_min = np.random.uniform(0.6, 0.9)
                    factor_max = np.random.uniform(1.1, 1.4)
                    df_aumentado = self._perturbar_velocidad(df_aumentado, factor_min, factor_max)
                
                elif estrategia == 'invertir_direccion':
                    df_aumentado = self._invertir_direccion(df_aumentado)
                
                elif estrategia == 'rotar':
                    angulo = np.random.uniform(0, 360)
                    df_aumentado = self._rotar_trayectoria(df_aumentado, angulo)
                
                elif estrategia == 'aumentar_objetos':
                    factor = np.random.uniform(1.2, 2.0)
                    df_aumentado = self._generar_variante_aumento_objetos(df_aumentado, factor)
                
                elif estrategia == 'mezclar_videos':
                    # Seleccionar un segundo video aleatorio diferente
                    nombres_restantes = [n for n in nombres_originales if n != nombre_base]
                    if nombres_restantes:
                        nombre_segundo = np.random.choice(nombres_restantes)
                        df_segundo = self.dataframes[nombre_segundo].copy()
                        proporcion = np.random.uniform(0.3, 0.7)
                        df_aumentado = self._mezclar_videos(df_aumentado, df_segundo, proporcion)
                
                elif estrategia == 'ruido_gaussiano':
                    nivel = np.random.uniform(0.02, 0.1)
                    df_aumentado = self._aplicar_ruido_gaussiano(df_aumentado, nivel)
                
                elif estrategia == 'fragmentar':
                    inicio = np.random.uniform(0.0, 0.3)
                    fin = np.random.uniform(0.7, 1.0)
                    df_aumentado = self._generar_fragmento(df_aumentado, inicio, fin)
            
            # Crear nombre para el archivo aumentado
            estrategias_str = '_'.join([e[:3] for e in estrategias_aplicar])
            nombre_aumentado = f"{nombre_base}_aug_{i+1}_{estrategias_str}"
            
            # Guardar archivo aumentado
            ruta_salida = os.path.join(self.output_directory, f"{nombre_aumentado}.csv")
            df_aumentado.to_csv(ruta_salida, index=False)
            
            archivos_generados += 1
            if archivos_generados % 10 == 0:
                print(f"  ✓ Generados {archivos_generados}/{num_aumentados} archivos")
        
        print(f"Aumento de datos completado. {archivos_generados} archivos generados en {self.output_directory}")
        return True
    
    def generar_dataset_balanceado(self, campo_balance='Patron_Movimiento', min_por_clase=30):
        """
        Genera un dataset balanceado asegurando un mínimo de ejemplos por clase
        
        Parámetros:
        -----------
        campo_balance : str
            Campo por el cual balancear (ej: 'Patron_Movimiento', 'Video')
        min_por_clase : int
            Mínimo de archivos a generar por cada clase
        """
        if not self.dataframes:
            if not self.cargar_datos():
                print("No se pudieron cargar los datos originales")
                return False
        
        # Analizar distribución actual
        distribucion = {}
        for nombre, df in self.dataframes.items():
            if campo_balance in df.columns:
                # Contar valores únicos en este archivo
                valores = df[campo_balance].value_counts().to_dict()
                
                for valor, conteo in valores.items():
                    if valor not in distribucion:
                        distribucion[valor] = []
                    
                    # Almacenar (nombre_archivo, conteo)
                    distribucion[valor].append((nombre, conteo))
        
        print(f"Distribución actual para '{campo_balance}':")
        for valor, archivos in distribucion.items():
            total = sum([conteo for _, conteo in archivos])
            print(f"  {valor}: {total} ejemplos en {len(archivos)} archivos")
        
        # Generar archivos aumentados para balancear
        for valor, archivos in distribucion.items():
            total_actual = sum([conteo for _, conteo in archivos])
            
            if total_actual < min_por_clase:
                a_generar = min_por_clase - total_actual
                print(f"\nGenerando {a_generar} ejemplos adicionales para '{valor}'")
                
                # Seleccionar archivos base para aumentar
                archivos_base = [nombre for nombre, _ in archivos]
                
                # Generar archivos
                generados = 0
                while generados < a_generar:
                    # Seleccionar archivo base aleatorio
                    nombre_base = np.random.choice(archivos_base)
                    df_base = self.dataframes[nombre_base].copy()
                    
                    # Aplicar estrategias aleatorias
                    df_aumentado = df_base.copy()
                    
                    # Aplicar 2-3 transformaciones aleatorias
                    transformaciones = [
                        self._interpolar_trayectoria,
                        self._perturbar_velocidad,
                        self._invertir_direccion,
                        self._rotar_trayectoria,
                        self._aplicar_ruido_gaussiano
                    ]
                    
                    num_trans = np.random.randint(2, 4)
                    trans_seleccionadas = np.random.choice(transformaciones, size=num_trans, replace=False)
                    
                    for trans_func in trans_seleccionadas:
                        if trans_func == self._interpolar_trayectoria:
                            df_aumentado = trans_func(df_aumentado, factor=np.random.uniform(1.5, 2.5))
                        elif trans_func == self._perturbar_velocidad:
                            df_aumentado = trans_func(df_aumentado, 
                                                     factor_min=np.random.uniform(0.7, 0.9),
                                                     factor_max=np.random.uniform(1.1, 1.3))
                        elif trans_func == self._rotar_trayectoria:
                            df_aumentado = trans_func(df_aumentado, angulo_grados=np.random.uniform(0, 360))
                        elif trans_func == self._aplicar_ruido_gaussiano:
                            df_aumentado = trans_func(df_aumentado, nivel_ruido=np.random.uniform(0.03, 0.08))
                        else:
                            df_aumentado = trans_func(df_aumentado)
                    
                    # Guardar archivo aumentado
                    nombre_aumentado = f"{nombre_base}_aug_bal_{valor.replace(' ', '_')}_{generados+1}"
                    ruta_salida = os.path.join(self.output_directory, f"{nombre_aumentado}.csv")
                    df_aumentado.to_csv(ruta_salida, index=False)
                    
                    generados += 1
                    if generados % 5 == 0:
                        print(f"  ✓ Generados {generados}/{a_generar} para '{valor}'")
        
        print("\nBalanceo de datos completado.")
        return True
    
    def analizar_dataset_aumentado(self):
        """
        Analiza el dataset original y aumentado para verificar distribuciones
        """
        # Verificar directorio aumentado
        csv_aumentados = glob.glob(os.path.join(self.output_directory, "*.csv"))
        
        if not csv_aumentados:
            print("No se encontraron archivos CSV aumentados para analizar")
            return
        
        print(f"\nAnalizando dataset aumentado ({len(csv_aumentados)} archivos)...")
        
        # Cargar y analizar archivos aumentados
        datos_aumentados = {}
        metadatos_aumentados = {}
        
        for csv_file in csv_aumentados:
            nombre = os.path.basename(csv_file).split('.')[0]
            try:
                df = pd.read_csv(csv_file)
                datos_aumentados[nombre] = df
                
                # Extraer metadatos
                metadatos_aumentados[nombre] = {
                    'duracion_frames': df['Frame'].max() - df['Frame'].min() + 1,
                    'objetos': df['Objeto'].nunique(),
                    'patrones': df['Patron_Movimiento'].value_counts().to_dict() if 'Patron_Movimiento' in df.columns else {}
                }
            except Exception as e:
                print(f"  ✗ Error al analizar {nombre}: {str(e)}")
        
        # Comparar métricas
        print("\nComparativa entre dataset original y aumentado:")
        
        # Total de frames
        frames_orig = sum([meta['duracion_frames'] for meta in self.metadata.values()])
        frames_aug = sum([meta['duracion_frames'] for meta in metadatos_aumentados.values()])
        
        print(f"  Frames totales: {frames_orig} → {frames_aug} ({frames_aug/frames_orig:.1f}x)")
        
        # Total de objetos
        objetos_orig = sum([meta['objetos'] for meta in self.metadata.values()])
        objetos_aug = sum([meta['objetos'] for meta in metadatos_aumentados.values()])
        
        print(f"  Objetos totales: {objetos_orig} → {objetos_aug} ({objetos_aug/objetos_orig:.1f}x)")
        
        # Distribución de patrones (si existe)
        patrones_orig = {}
        for meta in self.metadata.values():
            if 'patrones' in meta:
                for patron, conteo in meta['patrones'].items():
                    patrones_orig[patron] = patrones_orig.get(patron, 0) + conteo
        
        patrones_aug = {}
        for meta in metadatos_aumentados.values():
            if 'patrones' in meta:
                for patron, conteo in meta['patrones'].items():
                    patrones_aug[patron] = patrones_aug.get(patron, 0) + conteo
        
        if patrones_orig and patrones_aug:
            print("\nDistribución de patrones de movimiento:")
            print("Patrón\tOriginal\tAumentado\tFactor")
            print("-" * 50)
            for patron in sorted(set(list(patrones_orig.keys()) + list(patrones_aug.keys()))):
                orig = patrones_orig.get(patron, 0)
                aug = patrones_aug.get(patron, 0)
                factor = "N/A" if orig == 0 else f"{aug/orig:.1f}x"
                print(f"{patron}\t{orig}\t\t{aug}\t\t{factor}")
        
        return {
            'orig_files': len(self.dataframes),
            'aug_files': len(datos_aumentados),
            'orig_frames': frames_orig,
            'aug_frames': frames_aug,
            'orig_objects': objetos_orig,
            'aug_objects': objetos_aug,
            'orig_patterns': patrones_orig,
            'aug_patterns': patrones_aug
        }

    def visualizar_ejemplo_aumentado(self, nombre_original, nombre_aumentado=None):
        """
        Visualiza comparación entre un archivo original y su versión aumentada
        """
        import matplotlib.pyplot as plt
        
        if nombre_original not in self.dataframes:
            print(f"Error: No se encontró el archivo original '{nombre_original}'")
            return
        
        df_orig = self.dataframes[nombre_original]
        
        # Si no se especifica el aumentado, buscar uno que contenga el nombre original
        if nombre_aumentado is None:
            csv_aumentados = glob.glob(os.path.join(self.output_directory, f"{nombre_original}_aug_*.csv"))
            if not csv_aumentados:
                print(f"No se encontraron versiones aumentadas de '{nombre_original}'")
                return
            
            # Seleccionar el primer archivo aumentado encontrado
            nombre_aumentado = os.path.basename(csv_aumentados[0]).split('.')[0]
        
        # Cargar datos aumentados
        ruta_aumentado = os.path.join(self.output_directory, f"{nombre_aumentado}.csv")
        if not os.path.exists(ruta_aumentado):
            print(f"Error: No se encontró el archivo aumentado '{nombre_aumentado}'")
            return
        
        df_aug = pd.read_csv(ruta_aumentado)
        
        # Comprobar que existen las columnas necesarias
        if 'Centroide_X' not in df_orig.columns or 'Centroide_Y' not in df_orig.columns:
            print("Error: Los datos deben contener las columnas 'Centroide_X' y 'Centroide_Y'")
            return
        
        # Crear visualización
        plt.figure(figsize=(15, 7))
        
        # Gráfico 1: Original
        plt.subplot(1, 2, 1)
        for obj in df_orig['Objeto'].unique():
            datos_obj = df_orig[df_orig['Objeto'] == obj]
            plt.plot(datos_obj['Centroide_X'], datos_obj['Centroide_Y'], '-o', label=f"Obj {obj}")
        
        plt.title(f"Original: {nombre_original}")
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True, alpha=0.3)
        
        # Gráfico 2: Aumentado
        plt.subplot(1, 2, 2)
        for obj in df_aug['Objeto'].unique():
            datos_obj = df_aug[df_aug['Objeto'] == obj]
            plt.plot(datos_obj['Centroide_X'], datos_obj['Centroide_Y'], '-o', label=f"Obj {obj}")
        
        plt.title(f"Aumentado: {nombre_aumentado}")
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Guardar visualización
        nombre_img = f"comparacion_{nombre_original}_{nombre_aumentado}.png"
        plt.savefig(os.path.join(self.output_directory, nombre_img))
        print(f"Visualización guardada en: {os.path.join(self.output_directory, nombre_img)}")
        
        plt.close()

# Función principal de ejemplo
def main():
    # Configurar directorios
    directorio_csv = "./informacion/csv"  # Directorio con CSVs originales
    directorio_salida = "./informacion/csv_aumentados"  # Directorio para CSVs aumentados
    
    # Inicializar aumentador
    augmenter = MovementDataAugmenter(directorio_csv, directorio_salida)
    
    # Cargar datos
    if not augmenter.cargar_datos():
        print("Error: No se encontraron datos para aumentar.")
        return
    
    # Generar datos aumentados (ajustar según necesidades)
    augmenter.generar_dataset_aumentado(
        num_aumentados=200,  # Multiplicar original x4
        estrategias=[        # Usar todas las estrategias disponibles
            'interpolar', 
            'perturbar_velocidad', 
            'invertir_direccion',
            'rotar',
            'aumentar_objetos',
            'mezclar_videos',
            'ruido_gaussiano',
            'fragmentar'
        ]
    )
    
    # Asegurar balanceo de clases por patrones de movimiento
    augmenter.generar_dataset_balanceado(
        campo_balance='Patron_Movimiento',
        min_por_clase=50  # Asegurar al menos 50 ejemplos por patrón
    )
    
    # Analizar el dataset aumentado
    resultados = augmenter.analizar_dataset_aumentado()
    
    # Visualizar un ejemplo
    if len(augmenter.dataframes) > 0:
        nombre_ejemplo = list(augmenter.dataframes.keys())[0]
        augmenter.visualizar_ejemplo_aumentado(nombre_ejemplo)

if __name__ == "__main__":
    main()
    """ Módulo para técnicas de aumento de datos en análisis de movimiento y comportamiento.
Este script proporciona funciones para incrementar el volumen de datos disponibles
a partir de un conjunto limitado de archivos CSV de trayectorias.
"""
import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
import glob
import random
import math
import warnings
warnings.filterwarnings('ignore')

class MovementDataAugmenter:
    def __init__(self, csv_directory, output_directory):
        """
        Inicializa el aumentador de datos de movimiento
        
        Parámetros:
        -----------
        csv_directory : str
            Ruta al directorio que contiene los archivos CSV originales
        output_directory : str
            Ruta donde se guardarán los archivos CSV aumentados
        """
        self.csv_directory = csv_directory
        self.output_directory = output_directory
        self.dataframes = {}  # Almacenar DataFrames originales
        self.metadata = {}    # Almacenar metadatos (etiquetas, patrones, etc)
        
        # Crear directorio de salida si no existe
        os.makedirs(output_directory, exist_ok=True)
    
    def cargar_datos(self):
        """Carga todos los archivos CSV del directorio especificado"""
        csv_files = glob.glob(os.path.join(self.csv_directory, "*.csv"))
        
        if not csv_files:
            print(f"No se encontraron archivos CSV en {self.csv_directory}")
            return False
        
        print(f"Cargando {len(csv_files)} archivos CSV originales...")
        
        for csv_file in csv_files:
            nombre_video = os.path.basename(csv_file).split('.')[0]
            try:
                df = pd.read_csv(csv_file)
                self.dataframes[nombre_video] = df
                
                # Extraer metadatos básicos para cada archivo
                self.metadata[nombre_video] = {
                    'duracion_frames': df['Frame'].max() - df['Frame'].min() + 1,
                    'objetos': df['Objeto'].nunique(),
                    'patrones': df['Patron_Movimiento'].value_counts().to_dict() if 'Patron_Movimiento' in df.columns else {}
                }
                
                print(f"  ✓ Cargado {nombre_video}: {len(df)} registros")
            except Exception as e:
                print(f"  ✗ Error al cargar {nombre_video}: {str(e)}")
        
        print(f"Total de archivos cargados: {len(self.dataframes)}")
        return len(self.dataframes) > 0
    
    def _interpolar_trayectoria(self, df, factor=2.0):
        """
        Interpola puntos en la trayectoria para crear movimientos más suaves y duplicar datos
        
        Parámetros:
        -----------
        df : DataFrame
            DataFrame con datos de trayectoria
        factor : float
            Factor de interpolación (2.0 = duplicar puntos)
        """
        # Agrupar por objeto para procesar cada trayectoria por separado
        objetos = df['Objeto'].unique()
        frames = np.sort(df['Frame'].unique())
        
        # Nuevo DataFrame para almacenar resultados
        nuevos_registros = []
        
        for obj in objetos:
            # Extraer trayectoria de este objeto
            traj = df[df['Objeto'] == obj].sort_values('Frame')
            
            if len(traj) < 3:  # Necesitamos al menos 3 puntos para interpolar
                nuevos_registros.append(traj)
                continue
            
            # Crear nuevos frames interpolados
            frames_orig = traj['Frame'].values
            n_frames = len(frames_orig)
            frames_nuevos = np.linspace(frames_orig.min(), frames_orig.max(), 
                                        int(n_frames * factor))
            
            # Interpolar columnas numéricas
            columnas_numericas = traj.select_dtypes(include=['float64', 'int64']).columns
            columnas_numericas = [c for c in columnas_numericas if c != 'Frame' and c != 'Objeto']
            
            # Crear interpoladores para cada columna numérica
            interpoladores = {}
            for col in columnas_numericas:
                # Usar interpolación cúbica cuando hay suficientes puntos, lineal si no
                kind = 'cubic' if len(traj) >= 4 else 'linear'
                interpoladores[col] = interp1d(frames_orig, traj[col].values, 
                                              kind=kind, bounds_error=False, 
                                              fill_value='extrapolate')
            
            # Crear nuevos registros interpolados
            for frame_nuevo in frames_nuevos:
                nuevo_registro = {'Frame': frame_nuevo, 'Objeto': obj}
                
                for col in columnas_numericas:
                    nuevo_registro[col] = float(interpoladores[col](frame_nuevo))
                
                # Copiar columnas categóricas del frame más cercano
                frame_cercano = frames_orig[np.argmin(np.abs(frames_orig - frame_nuevo))]
                registro_cercano = traj[traj['Frame'] == frame_cercano].iloc[0]
                
                for col in traj.select_dtypes(include=['object']).columns:
                    nuevo_registro[col] = registro_cercano[col]
                
                nuevos_registros.append(nuevo_registro)
        
        return pd.DataFrame(nuevos_registros)
    
    def _perturbar_velocidad(self, df, factor_min=0.8, factor_max=1.2):
        """
        Perturba la velocidad y aceleración manteniendo la trayectoria
        
        Parámetros:
        -----------
        df : DataFrame
            DataFrame con datos de trayectoria
        factor_min, factor_max : float
            Rango de factores para perturbar la velocidad
        """
        # Crear copia para no modificar el original
        df_perturbado = df.copy()
        
        # Columnas a perturbar
        columnas_velocidad = ['Velocidad', 'Aceleracion', 'Desplazamiento']
        
        # Perturbar por objeto para mantener consistencia
        for obj in df['Objeto'].unique():
            # Generar factor de perturbación para este objeto
            factor = np.random.uniform(factor_min, factor_max)
            
            # Aplicar perturbación
            for col in columnas_velocidad:
                if col in df.columns:
                    mask = df_perturbado['Objeto'] == obj
                    df_perturbado.loc[mask, col] = df_perturbado.loc[mask, col] * factor
        
        return df_perturbado
    
    def _invertir_direccion(self, df):
        """
        Invierte la dirección del movimiento manteniendo el patrón
        """
        # Crear copia para no modificar el original
        df_invertido = df.copy()
        
        # Si existe la columna Dirección, invertir (añadir 180° módulo 360)
        if 'Direccion' in df.columns:
            df_invertido['Direccion'] = (df_invertido['Direccion'] + 180) % 360
        
        return df_invertido
    
    def _rotar_trayectoria(self, df, angulo_grados):
        """
        Rota la trayectoria un ángulo específico
        
        Parámetros:
        -----------
        df : DataFrame
            DataFrame con datos de trayectoria
        angulo_grados : float
            Ángulo de rotación en grados
        """
        # Crear copia para no modificar el original
        df_rotado = df.copy()
        
        # Convertir ángulo a radianes
        angulo_rad = np.radians(angulo_grados)
        
        # Matriz de rotación 2D
        cos_a = np.cos(angulo_rad)
        sin_a = np.sin(angulo_rad)
        
        # Rotar por objeto
        for obj in df['Objeto'].unique():
            mask = df_rotado['Objeto'] == obj
            
            # Asegurar que tenemos las coordenadas
            if 'Centroide_X' in df.columns and 'Centroide_Y' in df.columns:
                # Extraer coordenadas
                x = df_rotado.loc[mask, 'Centroide_X'].values
                y = df_rotado.loc[mask, 'Centroide_Y'].values
                
                # Calcular centro de la trayectoria
                cx = np.mean(x)
                cy = np.mean(y)
                
                # Trasladar al origen
                x_origen = x - cx
                y_origen = y - cy
                
                # Aplicar rotación
                x_rotado = x_origen * cos_a - y_origen * sin_a
                y_rotado = x_origen * sin_a + y_origen * cos_a
                
                # Trasladar de vuelta
                df_rotado.loc[mask, 'Centroide_X'] = x_rotado + cx
                df_rotado.loc[mask, 'Centroide_Y'] = y_rotado + cy
                
                # Actualizar dirección si existe
                if 'Direccion' in df.columns:
                    df_rotado.loc[mask, 'Direccion'] = (df_rotado.loc[mask, 'Direccion'] + angulo_grados) % 360
        
        return df_rotado
    
    def _generar_variante_aumento_objetos(self, df, factor=1.5):
        """
        Genera una variante con más objetos combinando trayectorias aleatorias del dataset
        
        Parámetros:
        -----------
        df : DataFrame
            DataFrame con datos de trayectoria
        factor : float
            Factor de aumento de objetos (1.5 = 50% más objetos)
        """
        df_aumentado = df.copy()
        
        # Obtener objetos originales
        objetos_orig = df['Objeto'].unique()
        num_objetos = len(objetos_orig)
        objetos_a_agregar = int(num_objetos * (factor - 1)) + 1
        
        # Crear nuevos IDs para los objetos a agregar
        ultimo_id = max([int(str(obj).replace('obj_', '')) 
                        if str(obj).startswith('obj_') else 0 
                        for obj in objetos_orig]) + 1
        
        # Agregar nuevos objetos
        for i in range(objetos_a_agregar):
            # Seleccionar un objeto aleatorio para clonar
            obj_orig = np.random.choice(objetos_orig)
            nuevo_obj_id = f"obj_{ultimo_id + i}"
            
            # Clonar datos
            datos_obj = df[df['Objeto'] == obj_orig].copy()
            datos_obj['Objeto'] = nuevo_obj_id
            
            # Modificar posición para evitar superposición exacta
            offset_x = np.random.uniform(-50, 50)
            offset_y = np.random.uniform(-50, 50)
            
            if 'Centroide_X' in datos_obj.columns and 'Centroide_Y' in datos_obj.columns:
                datos_obj['Centroide_X'] += offset_x
                datos_obj['Centroide_Y'] += offset_y
            
            # Perturbar velocidad y otras métricas
            datos_obj = self._perturbar_velocidad(datos_obj)
            
            # Añadir al dataframe
            df_aumentado = pd.concat([df_aumentado, datos_obj], ignore_index=True)
        
        return df_aumentado
    
    def _mezclar_videos(self, df1, df2, proporcion=0.5):
        """
        Crea un nuevo video mezclando objetos de dos videos diferentes
        
        Parámetros:
        -----------
        df1, df2 : DataFrame
            DataFrames de dos videos a mezclar
        proporcion : float
            Proporción de objetos a tomar del segundo video (0.5 = 50%)
        """
        # Crear copia del primer video
        df_mezclado = df1.copy()
        
        # Extraer objetos del segundo video
        objetos_df2 = df2['Objeto'].unique()
        num_objetos_a_tomar = max(1, int(len(objetos_df2) * proporcion))
        objetos_seleccionados = np.random.choice(objetos_df2, 
                                                size=num_objetos_a_tomar, 
                                                replace=False)
        
        # Crear nuevos IDs para evitar colisiones
        objetos_df1 = df1['Objeto'].unique()
        ultimo_id = max([int(str(obj).replace('obj_', '')) 
                        if str(obj).startswith('obj_') else 0 
                        for obj in objetos_df1]) + 1
        
        # Añadir objetos seleccionados con nuevos IDs
        for i, obj in enumerate(objetos_seleccionados):
            datos_obj = df2[df2['Objeto'] == obj].copy()
            
            # Ajustar IDs
            nuevo_obj_id = f"obj_{ultimo_id + i}"
            datos_obj['Objeto'] = nuevo_obj_id
            
            # Ajustar posiciones si es necesario
            if 'Centroide_X' in datos_obj.columns and 'Centroide_Y' in datos_obj.columns:
                # Calcular centroides medios de ambos dataframes
                centro1_x = df1['Centroide_X'].mean() if 'Centroide_X' in df1.columns else 0
                centro1_y = df1['Centroide_Y'].mean() if 'Centroide_Y' in df1.columns else 0
                
                centro2_x = df2['Centroide_X'].mean() if 'Centroide_X' in df2.columns else 0
                centro2_y = df2['Centroide_Y'].mean() if 'Centroide_Y' in df2.columns else 0
                
                # Calcular offset
                offset_x = centro1_x - centro2_x
                offset_y = centro1_y - centro2_y
                
                # Aplicar ajuste y añadir desplazamiento aleatorio
                datos_obj['Centroide_X'] += offset_x + np.random.uniform(-30, 30)
                datos_obj['Centroide_Y'] += offset_y + np.random.uniform(-30, 30)
            
            # Añadir al dataframe mezclado
            df_mezclado = pd.concat([df_mezclado, datos_obj], ignore_index=True)
        
        return df_mezclado
    
    def _aplicar_ruido_gaussiano(self, df, nivel_ruido=0.05):
        """
        Aplica ruido gaussiano a variables numéricas para simular variaciones
        
        Parámetros:
        -----------
        df : DataFrame
            DataFrame con datos de trayectoria
        nivel_ruido : float
            Intensidad del ruido (desviación estándar como proporción de la media)
        """
        # Crear copia para no modificar el original
        df_ruidoso = df.copy()
        
        # Identificar columnas numéricas
        columnas_num = df.select_dtypes(include=['float64', 'int64']).columns
        columnas_num = [c for c in columnas_num if c != 'Frame' and c != 'Objeto']
        
        # Aplicar ruido por columna
        for col in columnas_num:
            # Calcular nivel de ruido específico para esta columna
            media = np.abs(df[col].mean())
            sigma = media * nivel_ruido if media > 0 else 0.1
            
            # Generar ruido
            ruido = np.random.normal(0, sigma, size=len(df_ruidoso))
            
            # Aplicar ruido
            df_ruidoso[col] = df_ruidoso[col] + ruido
        
        return df_ruidoso
    
    def _generar_fragmento(self, df, inicio_rel=0.0, fin_rel=1.0):
        """
        Genera un fragmento de video basado en proporciones relativas del original
        
        Parámetros:
        -----------
        df : DataFrame
            DataFrame con datos de trayectoria
        inicio_rel, fin_rel : float
            Proporciones relativas de inicio y fin (0.0 a 1.0)
        """
        # Obtener rango de frames
        frame_min = df['Frame'].min()
        frame_max = df['Frame'].max()
        rango_frames = frame_max - frame_min
        
        # Calcular frames de inicio y fin
        frame_inicio = int(frame_min + rango_frames * inicio_rel)
        frame_fin = int(frame_min + rango_frames * fin_rel)
        
        # Extraer fragmento
        fragmento = df[(df['Frame'] >= frame_inicio) & (df['Frame'] <= frame_fin)].copy()
        
        # Reindexar frames para que comiencen desde 0
        fragmento['Frame'] = fragmento['Frame'] - frame_inicio
        
        return fragmento
    
    def generar_dataset_aumentado(self, num_aumentados=100, estrategias=None):
        """
        Genera un dataset aumentado aplicando múltiples estrategias
        
        Parámetros:
        -----------
        num_aumentados : int
            Número de archivos aumentados a generar
        estrategias : list
            Lista de estrategias a aplicar. Si es None, usa todas.
        """
        if not self.dataframes:
            if not self.cargar_datos():
                print("No se pudieron cargar los datos originales")
                return False
        
        # Listar todas las estrategias disponibles
        todas_estrategias = [
            'interpolar',           # Interpolar puntos intermedios
            'perturbar_velocidad',  # Cambiar velocidad
            'invertir_direccion',   # Invertir dirección
            'rotar',                # Rotar trayectoria
            'aumentar_objetos',     # Añadir más objetos
            'mezclar_videos',       # Mezclar objetos de diferentes videos
            'ruido_gaussiano',      # Añadir ruido aleatorio
            'fragmentar'            # Crear fragmentos de videos
        ]
        
        # Si no se especifican estrategias, usar todas
        if estrategias is None:
            estrategias = todas_estrategias
        
        print(f"Generando {num_aumentados} archivos aumentados usando estrategias: {estrategias}")
        
        # Obtener lista de archivos originales
        nombres_originales = list(self.dataframes.keys())
        
        # Contadores para seguimiento
        archivos_generados = 0
        
        # Generar archivos aumentados
        for i in range(num_aumentados):
            # Seleccionar archivo base aleatorio
            nombre_base = np.random.choice(nombres_originales)
            df_base = self.dataframes[nombre_base].copy()
            
            # Inicializar DataFrame aumentado
            df_aumentado = df_base.copy()
            
            # Seleccionar estrategias aleatorias a aplicar (2-4 estrategias)
            num_estrategias = np.random.randint(2, min(5, len(estrategias) + 1))
            estrategias_aplicar = np.random.choice(estrategias, 
                                                  size=num_estrategias, 
                                                  replace=False)
            
            # Aplicar cada estrategia seleccionada
            for estrategia in estrategias_aplicar:
                if estrategia == 'interpolar':
                    factor = np.random.uniform(1.5, 3.0)
                    df_aumentado = self._interpolar_trayectoria(df_aumentado, factor)
                
                elif estrategia == 'perturbar_velocidad':
                    factor_min = np.random.uniform(0.6, 0.9)
                    factor_max = np.random.uniform(1.1, 1.4)
                    df_aumentado = self._perturbar_velocidad(df_aumentado, factor_min, factor_max)
                
                elif estrategia == 'invertir_direccion':
                    df_aumentado = self._invertir_direccion(df_aumentado)
                
                elif estrategia == 'rotar':
                    angulo = np.random.uniform(0, 360)
                    df_aumentado = self._rotar_trayectoria(df_aumentado, angulo)
                
                elif estrategia == 'aumentar_objetos':
                    factor = np.random.uniform(1.2, 2.0)
                    df_aumentado = self._generar_variante_aumento_objetos(df_aumentado, factor)
                
                elif estrategia == 'mezclar_videos':
                    # Seleccionar un segundo video aleatorio diferente
                    nombres_restantes = [n for n in nombres_originales if n != nombre_base]
                    if nombres_restantes:
                        nombre_segundo = np.random.choice(nombres_restantes)
                        df_segundo = self.dataframes[nombre_segundo].copy()
                        proporcion = np.random.uniform(0.3, 0.7)
                        df_aumentado = self._mezclar_videos(df_aumentado, df_segundo, proporcion)
                
                elif estrategia == 'ruido_gaussiano':
                    nivel = np.random.uniform(0.02, 0.1)
                    df_aumentado = self._aplicar_ruido_gaussiano(df_aumentado, nivel)
                
                elif estrategia == 'fragmentar':
                    inicio = np.random.uniform(0.0, 0.3)
                    fin = np.random.uniform(0.7, 1.0)
                    df_aumentado = self._generar_fragmento(df_aumentado, inicio, fin)
            
            # Crear nombre para el archivo aumentado
            estrategias_str = '_'.join([e[:3] for e in estrategias_aplicar])
            nombre_aumentado = f"{nombre_base}_aug_{i+1}_{estrategias_str}"
            
            # Guardar archivo aumentado
            ruta_salida = os.path.join(self.output_directory, f"{nombre_aumentado}.csv")
            df_aumentado.to_csv(ruta_salida, index=False)
            
            archivos_generados += 1
            if archivos_generados % 10 == 0:
                print(f"  ✓ Generados {archivos_generados}/{num_aumentados} archivos")
        
        print(f"Aumento de datos completado. {archivos_generados} archivos generados en {self.output_directory}")
        return True
    
    def generar_dataset_balanceado(self, campo_balance='Patron_Movimiento', min_por_clase=30):
        """
        Genera un dataset balanceado asegurando un mínimo de ejemplos por clase
        
        Parámetros:
        -----------
        campo_balance : str
            Campo por el cual balancear (ej: 'Patron_Movimiento', 'Video')
        min_por_clase : int
            Mínimo de archivos a generar por cada clase
        """
        if not self.dataframes:
            if not self.cargar_datos():
                print("No se pudieron cargar los datos originales")
                return False
        
        # Analizar distribución actual
        distribucion = {}
        for nombre, df in self.dataframes.items():
            if campo_balance in df.columns:
                # Contar valores únicos en este archivo
                valores = df[campo_balance].value_counts().to_dict()
                
                for valor, conteo in valores.items():
                    if valor not in distribucion:
                        distribucion[valor] = []
                    
                    # Almacenar (nombre_archivo, conteo)
                    distribucion[valor].append((nombre, conteo))
        
        print(f"Distribución actual para '{campo_balance}':")
        for valor, archivos in distribucion.items():
            total = sum([conteo for _, conteo in archivos])
            print(f"  {valor}: {total} ejemplos en {len(archivos)} archivos")
        
        # Generar archivos aumentados para balancear
        for valor, archivos in distribucion.items():
            total_actual = sum([conteo for _, conteo in archivos])
            
            if total_actual < min_por_clase:
                a_generar = min_por_clase - total_actual
                print(f"\nGenerando {a_generar} ejemplos adicionales para '{valor}'")
                
                # Seleccionar archivos base para aumentar
                archivos_base = [nombre for nombre, _ in archivos]
                
                # Generar archivos
                generados = 0
                while generados < a_generar:
                    # Seleccionar archivo base aleatorio
                    nombre_base = np.random.choice(archivos_base)
                    df_base = self.dataframes[nombre_base].copy()
                    
                    # Aplicar estrategias aleatorias
                    df_aumentado = df_base.copy()
                    
                    # Aplicar 2-3 transformaciones aleatorias
                    transformaciones = [
                        self._interpolar_trayectoria,
                        self._perturbar_velocidad,
                        self._invertir_direccion,
                        self._rotar_trayectoria,
                        self._aplicar_ruido_gaussiano
                    ]
                    
                    num_trans = np.random.randint(2, 4)
                    trans_seleccionadas = np.random.choice(transformaciones, size=num_trans, replace=False)
                    
                    for trans_func in trans_seleccionadas:
                        if trans_func == self._interpolar_trayectoria:
                            df_aumentado = trans_func(df_aumentado, factor=np.random.uniform(1.5, 2.5))
                        elif trans_func == self._perturbar_velocidad:
                            df_aumentado = trans_func(df_aumentado, 
                                                     factor_min=np.random.uniform(0.7, 0.9),
                                                     factor_max=np.random.uniform(1.1, 1.3))
                        elif trans_func == self._rotar_trayectoria:
                            df_aumentado = trans_func(df_aumentado, angulo_grados=np.random.uniform(0, 360))
                        elif trans_func == self._aplicar_ruido_gaussiano:
                            df_aumentado = trans_func(df_aumentado, nivel_ruido=np.random.uniform(0.03, 0.08))
                        else:
                            df_aumentado = trans_func(df_aumentado)
                    
                    # Guardar archivo aumentado
                    nombre_aumentado = f"{nombre_base}_aug_bal_{valor.replace(' ', '_')}_{generados+1}"
                    ruta_salida = os.path.join(self.output_directory, f"{nombre_aumentado}.csv")
                    df_aumentado.to_csv(ruta_salida, index=False)
                    
                    generados += 1
                    if generados % 5 == 0:
                        print(f"  ✓ Generados {generados}/{a_generar} para '{valor}'")
        
        print("\nBalanceo de datos completado.")
        return True
    
    def analizar_dataset_aumentado(self):
        """
        Analiza el dataset original y aumentado para verificar distribuciones
        """
        # Verificar directorio aumentado
        csv_aumentados = glob.glob(os.path.join(self.output_directory, "*.csv"))
        
        if not csv_aumentados:
            print("No se encontraron archivos CSV aumentados para analizar")
            return
        
        print(f"\nAnalizando dataset aumentado ({len(csv_aumentados)} archivos)...")
        
        # Cargar y analizar archivos aumentados
        datos_aumentados = {}
        metadatos_aumentados = {}
        
        for csv_file in csv_aumentados:
            nombre = os.path.basename(csv_file).split('.')[0]
            try:
                df = pd.read_csv(csv_file)
                datos_aumentados[nombre] = df
                
                # Extraer metadatos
                metadatos_aumentados[nombre] = {
                    'duracion_frames': df['Frame'].max() - df['Frame'].min() + 1,
                    'objetos': df['Objeto'].nunique(),
                    'patrones': df['Patron_Movimiento'].value_counts().to_dict() if 'Patron_Movimiento' in df.columns else {}
                }
            except Exception as e:
                print(f"  ✗ Error al analizar {nombre}: {str(e)}")
        
        # Comparar métricas
        print("\nComparativa entre dataset original y aumentado:")
        
        # Total de frames
        frames_orig = sum([meta['duracion_frames'] for meta in self.metadata.values()])
        frames_aug = sum([meta['duracion_frames'] for meta in metadatos_aumentados.values()])
        
        print(f"  Frames totales: {frames_orig} → {frames_aug} ({frames_aug/frames_orig:.1f}x)")
        
        # Total de objetos
        objetos_orig = sum([meta['objetos'] for meta in self.metadata.values()])
        objetos_aug = sum([meta['objetos'] for meta in metadatos_aumentados.values()])
        
        print(f"  Objetos totales: {objetos_orig} → {objetos_aug} ({objetos_aug/objetos_orig:.1f}x)")
        
        # Distribución de patrones (si existe)
        patrones_orig = {}
        for meta in self.metadata.values():
            if 'patrones' in meta:
                for patron, conteo in meta['patrones'].items():
                    patrones_orig[patron] = patrones_orig.get(patron, 0) + conteo
        
        patrones_aug = {}