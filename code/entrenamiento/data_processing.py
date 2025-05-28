"""
Procesador de datos mejorado que integra limpieza automática y soporte para datos originales.
Incluye tanto datos sin procesar como datos agregados por video, objeto y frame.
Versión corregida con mejor manejo de errores y robustez.
"""
import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from collections import Counter
import warnings
import json
from pathlib import Path
warnings.filterwarnings('ignore')

class EnhancedDataProcessor:
    def __init__(self, clean_data_dir='./cleaned_data/', output_dir='./processed_data/', 
                 use_original_data=True, use_aggregated_data=True):
        """
        Inicializa el procesador mejorado.
        
        Parámetros:
        -----------
        clean_data_dir : str
            Directorio con datos ya limpiados
        output_dir : str
            Directorio de salida para datos procesados
        use_original_data : bool
            Si procesar datos originales (sin agregación)
        use_aggregated_data : bool  
            Si crear datos agregados (video, objeto, frame)
        """
        self.clean_data_dir = Path(clean_data_dir)
        self.output_dir = Path(output_dir)
        self.use_original_data = use_original_data
        self.use_aggregated_data = use_aggregated_data
        
        self.classes = ['normal', 'forcejeo', 'merodeo', 'sospechoso']
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Inicializar atributos
        self.combined_data = None
        self.video_info = None
        
        # Crear directorios
        self._create_directories()
        
        print(f"🔧 Procesador mejorado inicializado:")
        print(f"   Datos limpios: {self.clean_data_dir}")
        print(f"   Salida: {self.output_dir}")
        print(f"   Datos originales: {'✅' if use_original_data else '❌'}")
        print(f"   Datos agregados: {'✅' if use_aggregated_data else '❌'}")
    
    def _create_directories(self):
        """Crea directorios necesarios."""
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            if self.use_aggregated_data:
                (self.output_dir / 'lstm_data').mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"⚠️ Error creando directorios: {e}")
            raise
    
    def load_cleaned_data(self):
        """
        Carga todos los datos ya limpiados automáticamente.
        """
        print("\n" + "="*60)
        print("CARGANDO DATOS LIMPIADOS AUTOMÁTICAMENTE")
        print("="*60)
        
        # Primero intentar cargar archivo combinado
        combined_file = self.clean_data_dir / 'all_data_cleaned.csv'
        
        if combined_file.exists():
            print("✅ Cargando archivo de datos combinados limpiados...")
            try:
                combined_data = pd.read_csv(combined_file)
                video_info = self._extract_video_info(combined_data)
            except Exception as e:
                print(f"❌ Error cargando archivo combinado: {e}")
                return self._load_individual_files()
        else:
            return self._load_individual_files()
        
        # Validar datos cargados
        if combined_data.empty:
            raise ValueError("Los datos cargados están vacíos")
        
        # Procesar datos
        combined_data, video_info = self._process_loaded_data(combined_data, video_info)
        
        return combined_data, video_info
    
    def _load_individual_files(self):
        """Carga archivos individuales por clase."""
        print("📂 Cargando datos limpiados por clase...")
        all_data = []
        video_info = []
        
        for class_name in self.classes:
            class_dir = self.clean_data_dir / class_name
            
            if not class_dir.exists():
                print(f"⚠️ Carpeta {class_dir} no encontrada")
                continue
            
            csv_files = list(class_dir.glob("clean_*.csv"))
            
            if not csv_files:
                print(f"⚠️ No se encontraron archivos limpios en {class_dir}")
                continue
            
            print(f"📁 Clase '{class_name}': {len(csv_files)} archivos limpios")
            
            for csv_file in csv_files:
                try:
                    video_name = csv_file.stem.replace('clean_', '')
                    df = pd.read_csv(csv_file)
                    
                    # Validar archivo
                    if df.empty:
                        print(f"  ⚠️ {video_name}: Archivo vacío")
                        continue
                    
                    # Asegurar columnas necesarias
                    df = self._ensure_required_columns(df, class_name, video_name)
                    
                    all_data.append(df)
                    
                    # Información del video
                    video_info.append(self._extract_single_video_info(df, video_name, class_name))
                    
                    print(f"  ✅ {video_name}: {len(df)} registros")
                    
                except Exception as e:
                    print(f"  ❌ Error en {csv_file}: {str(e)}")
        
        if not all_data:
            raise ValueError("No se pudieron cargar datos limpios")
        
        combined_data = pd.concat(all_data, ignore_index=True)
        video_info_df = pd.DataFrame(video_info)
        
        return self._process_loaded_data(combined_data, video_info_df)
    
    def _ensure_required_columns(self, df, class_name, video_name):
        """Asegura que el DataFrame tenga las columnas necesarias."""
        if 'clase' not in df.columns:
            df['clase'] = class_name
        if 'video_id' not in df.columns:
            df['video_id'] = video_name
        return df
    
    def _extract_video_info(self, combined_data):
        """Extrae información de videos desde datos combinados."""
        video_info = []
        for video_id in combined_data['video_id'].unique():
            video_data = combined_data[combined_data['video_id'] == video_id]
            video_info.append(self._extract_single_video_info(video_data, video_id))
        
        return pd.DataFrame(video_info)
    
    def _extract_single_video_info(self, video_data, video_id, class_name=None):
        """Extrae información de un solo video."""
        if class_name is None:
            class_name = video_data['clase'].iloc[0] if 'clase' in video_data.columns else 'unknown'
        
        # Calcular métricas de manera segura
        n_frames = video_data['Frame'].nunique() if 'Frame' in video_data.columns else len(video_data)
        n_objetos = video_data['Objeto'].nunique() if 'Objeto' in video_data.columns else 1
        
        if 'Frame' in video_data.columns and not video_data['Frame'].empty:
            duracion_aprox = video_data['Frame'].max() - video_data['Frame'].min() + 1
        else:
            duracion_aprox = 1
        
        return {
            'video_id': video_id,
            'clase': class_name,
            'n_frames': n_frames,
            'n_objetos': n_objetos,
            'n_registros': len(video_data),
            'duracion_aprox': duracion_aprox
        }
    
    def _process_loaded_data(self, combined_data, video_info):
        """Procesa los datos cargados (normalización, encoding, etc.)."""
        # Normalizar nombres de clases
        class_mapping = {'sospechoso': 'merodeo'}
        combined_data['clase'] = combined_data['clase'].replace(class_mapping)
        video_info['clase'] = video_info['clase'].replace(class_mapping)
        
        # Validar que tenemos clases
        unique_classes = combined_data['clase'].unique()
        if len(unique_classes) == 0:
            raise ValueError("No se encontraron clases en los datos")
        
        # Codificar clases
        self.label_encoder.fit(unique_classes)
        combined_data['clase_encoded'] = self.label_encoder.transform(combined_data['clase'])
        video_info['clase_encoded'] = self.label_encoder.transform(video_info['clase'])
        
        # Almacenar en atributos de clase
        self.combined_data = combined_data
        self.video_info = video_info
        
        # Mostrar estadísticas
        self._print_data_statistics()
        
        return combined_data, video_info
    
    def _print_data_statistics(self):
        """Imprime estadísticas de los datos cargados."""
        print(f"\n📊 ESTADÍSTICAS DE DATOS CARGADOS (YA LIMPIADOS):")
        print(f"  Total de registros: {len(self.combined_data):,}")
        print(f"  Total de videos: {len(self.video_info)}")
        
        # Distribución por clase
        print(f"\n📈 Distribución por clase:")
        class_dist = self.video_info['clase'].value_counts()
        for clase, count in class_dist.items():
            percentage = (count / len(self.video_info)) * 100
            print(f"  {clase}: {count} videos ({percentage:.1f}%)")
        
        # Verificar calidad de datos limpios
        numeric_cols = self.combined_data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            # Verificar NaN e infinitos
            nan_count = self.combined_data[numeric_cols].isnull().sum().sum()
            inf_count = np.isinf(self.combined_data[numeric_cols].select_dtypes(include=[np.number])).sum().sum()
            
            print(f"\n✅ CALIDAD DE DATOS LIMPIOS:")
            print(f"  NaN encontrados: {nan_count} {'✅' if nan_count == 0 else '⚠️'}")
            print(f"  Infinitos encontrados: {inf_count} {'✅' if inf_count == 0 else '⚠️'}")
            
            if nan_count > 0 or inf_count > 0:
                print("⚠️ Los datos requieren limpieza adicional")
            else:
                print("✅ Datos perfectamente limpios y listos para ML")
        else:
            print("⚠️ No se encontraron columnas numéricas")
    
    def process_original_data(self):
        """
        Procesa datos originales sin agregación.
        Mantiene la granularidad máxima de los datos.
        """
        if not self.use_original_data:
            print("⚠️ Procesamiento de datos originales deshabilitado")
            return None, None
        
        print(f"\n📊 PROCESANDO DATOS ORIGINALES (SIN AGREGACIÓN)...")
        
        if self.combined_data is None:
            raise ValueError("Datos no cargados. Ejecute load_cleaned_data() primero.")
        
        # Usar directamente los datos combinados limpios
        original_data = self.combined_data.copy()
        
        # Identificar características numéricas
        exclude_cols = ['video_id', 'clase', 'clase_encoded']
        numeric_cols = original_data.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        if len(feature_cols) == 0:
            print("⚠️ No se encontraron características numéricas")
            return original_data, []
        
        print(f"  ✅ Datos originales preparados:")
        print(f"     Registros: {len(original_data)}")
        print(f"     Características: {len(feature_cols)}")
        print(f"     Características incluidas: {feature_cols[:10]}...")
        
        return original_data, feature_cols
    
    def create_features_by_level(self):
        """
        Crea características agregadas por video, objeto y frame.
        """
        if not self.use_aggregated_data:
            print("⚠️ Datos agregados deshabilitados")
            return None, None, None
        
        if self.combined_data is None:
            raise ValueError("Datos no cargados. Ejecute load_cleaned_data() primero.")
        
        print(f"\n🔧 CREANDO CARACTERÍSTICAS AGREGADAS...")
        
        # Características base para agregación
        numeric_features = self._get_numeric_features_for_aggregation()
        
        if len(numeric_features) == 0:
            print("⚠️ No hay características numéricas para agregación")
            return None, None, None
        
        print(f"  Características disponibles para agregación: {len(numeric_features)}")
        
        # 1. NIVEL VIDEO
        video_features = self._create_video_features(numeric_features)
        
        # 2. NIVEL OBJETO
        object_features = self._create_object_features(numeric_features)
        
        # 3. NIVEL FRAME
        frame_features = self._create_frame_features(numeric_features)
        
        print(f"  ✅ Características agregadas creadas:")
        print(f"    Video: {len(video_features) if video_features is not None else 0} filas")
        print(f"    Objeto: {len(object_features) if object_features is not None else 0} filas")
        print(f"    Frame: {len(frame_features) if frame_features is not None else 0} filas")
        
        return video_features, object_features, frame_features
    
    def _get_numeric_features_for_aggregation(self):
        """Obtiene características numéricas válidas para agregación."""
        numeric_features = []
        exclude_cols = ['Frame', 'Objeto', 'clase_encoded', 'video_id']
        
        for col in self.combined_data.columns:
            if (self.combined_data[col].dtype in ['float64', 'int64'] and 
                col not in exclude_cols):
                # Verificar que la columna no esté completamente vacía o sea constante
                if not self.combined_data[col].isnull().all() and self.combined_data[col].nunique() > 1:
                    numeric_features.append(col)
        
        return numeric_features[:20]  # Limitar para evitar problemas de memoria
    
    def _create_video_features(self, numeric_features):
        """Crea características agregadas por video."""
        print(f"  Creando características por video...")
        
        try:
            # Preparar agregaciones de manera más robusta
            agg_dict = {}
            
            # Agregar características numéricas
            for feat in numeric_features[:15]:  # Limitar para evitar errores
                agg_dict[feat] = ['mean', 'max', 'std', 'min']
            
            # Agregar metadatos
            if 'Frame' in self.combined_data.columns:
                agg_dict['Frame'] = 'max'
            if 'Objeto' in self.combined_data.columns:
                agg_dict['Objeto'] = 'nunique'
            
            agg_dict['clase'] = 'first'
            agg_dict['clase_encoded'] = 'first'
            
            video_features = self.combined_data.groupby('video_id').agg(agg_dict)
            
            # Aplanar nombres de columnas de manera más robusta
            new_columns = []
            for col in video_features.columns:
                if isinstance(col, tuple):
                    new_columns.append('_'.join(map(str, col)).strip('_'))
                else:
                    new_columns.append(str(col))
            
            video_features.columns = new_columns
            video_features = video_features.reset_index()
            
            return video_features
            
        except Exception as e:
            print(f"    ❌ Error creando características por video: {e}")
            return None
    
    def _create_object_features(self, numeric_features):
        """Crea características agregadas por objeto."""
        print(f"  Creando características por objeto...")
        
        if 'Objeto' not in self.combined_data.columns:
            print("    ⚠️ Columna 'Objeto' no encontrada, saltando nivel objeto")
            return None
        
        try:
            agg_dict = {}
            
            # Agregar características numéricas
            for feat in numeric_features[:15]:
                agg_dict[feat] = ['mean', 'max', 'std']
            
            # Agregar metadatos
            if 'Frame' in self.combined_data.columns:
                agg_dict['Frame'] = ['min', 'max', 'count']
            
            agg_dict['clase'] = 'first'
            agg_dict['clase_encoded'] = 'first'
            
            object_features = self.combined_data.groupby(['video_id', 'Objeto']).agg(agg_dict)
            
            # Aplanar nombres de columnas
            new_columns = []
            for col in object_features.columns:
                if isinstance(col, tuple):
                    new_columns.append('_'.join(map(str, col)).strip('_'))
                else:
                    new_columns.append(str(col))
            
            object_features.columns = new_columns
            object_features = object_features.reset_index()
            
            return object_features
            
        except Exception as e:
            print(f"    ❌ Error creando características por objeto: {e}")
            return None
    
    def _create_frame_features(self, numeric_features):
        """Crea características agregadas por frame."""
        print(f"  Creando características por frame...")
        
        if 'Frame' not in self.combined_data.columns:
            print("    ⚠️ Columna 'Frame' no encontrada, saltando nivel frame")
            return None
        
        try:
            agg_dict = {}
            
            # Agregar características numéricas
            for feat in numeric_features[:15]:
                agg_dict[feat] = ['mean', 'max']
            
            # Agregar metadatos
            if 'Objeto' in self.combined_data.columns:
                agg_dict['Objeto'] = 'nunique'
            
            agg_dict['clase'] = 'first'
            agg_dict['clase_encoded'] = 'first'
            
            frame_features = self.combined_data.groupby(['video_id', 'Frame']).agg(agg_dict)
            
            # Aplanar nombres de columnas
            new_columns = []
            for col in frame_features.columns:
                if isinstance(col, tuple):
                    new_columns.append('_'.join(map(str, col)).strip('_'))
                else:
                    new_columns.append(str(col))
            
            frame_features.columns = new_columns
            frame_features = frame_features.reset_index()
            
            return frame_features
            
        except Exception as e:
            print(f"    ❌ Error creando características por frame: {e}")
            return None
    
    def create_lstm_sequences(self):
        """
        Crea secuencias temporales para LSTM si se requieren datos agregados.
        """
        if not self.use_aggregated_data:
            print("⚠️ LSTM deshabilitado (datos agregados requeridos)")
            return None, None
        
        if self.combined_data is None:
            raise ValueError("Datos no cargados. Ejecute load_cleaned_data() primero.")
        
        print(f"\n🕐 CREANDO SECUENCIAS TEMPORALES PARA LSTM...")
        
        sequences = {}
        sequence_info = []
        
        # Identificar características para secuencias
        numeric_cols = self.combined_data.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols 
                       if col not in ['Frame', 'Objeto', 'clase_encoded']]
        
        if len(feature_cols) == 0:
            print("❌ No hay características numéricas para LSTM")
            return None, None
        
        available_features = feature_cols[:10]  # Limitar a 10 características
        print(f"  Usando {len(available_features)} características para secuencias")
        
        for video_id in self.combined_data['video_id'].unique():
            try:
                video_data = self.combined_data[self.combined_data['video_id'] == video_id]
                
                # Crear secuencia basada en frames o índice temporal
                sequence = self._create_video_sequence(video_data, available_features)
                
                if sequence is not None and len(sequence) > 0 and not np.isnan(sequence).all():
                    sequences[video_id] = sequence
                    
                    sequence_info.append({
                        'video_id': video_id,
                        'sequence_length': len(sequence),
                        'num_features': sequence.shape[1] if len(sequence.shape) > 1 else 1,
                        'clase': video_data['clase'].iloc[0],
                        'clase_encoded': video_data['clase_encoded'].iloc[0]
                    })
                
            except Exception as e:
                print(f"    ⚠️ Error procesando {video_id}: {str(e)}")
        
        if sequences:
            avg_length = np.mean([len(seq) for seq in sequences.values()])
            print(f"  ✅ {len(sequences)} secuencias creadas")
            print(f"  Longitud promedio: {avg_length:.1f}")
        else:
            print("❌ No se pudieron crear secuencias")
        
        return sequences, pd.DataFrame(sequence_info) if sequence_info else None
    
    def _create_video_sequence(self, video_data, features):
        """Crea secuencia temporal para un video específico."""
        try:
            if 'Frame' in video_data.columns and video_data['Frame'].nunique() > 1:
                # Agrupar por frame
                frame_stats = video_data.groupby('Frame')[features].mean()
                # Rellenar NaN con interpolación
                frame_stats = frame_stats.interpolate().fillna(method='bfill').fillna(method='ffill')
            else:
                # Si no hay Frame, usar ventana deslizante
                video_features = video_data[features].fillna(0)
                if len(video_features) > 5:
                    frame_stats = video_features.rolling(window=5, min_periods=1).mean()
                else:
                    frame_stats = video_features
            
            sequence = frame_stats.values
            
            # Validar secuencia
            if len(sequence) == 0 or np.isnan(sequence).all():
                return None
            
            return sequence
            
        except Exception as e:
            print(f"      Error creando secuencia: {e}")
            return None
    
    def split_data_optimized(self, test_size=0.25, val_size=0.15, random_state=42):
        """
        División optimizada con datos reservados para evaluación final.
        """
        if self.video_info is None:
            raise ValueError("Información de video no disponible. Ejecute load_cleaned_data() primero.")
        
        print(f"\n✂️ DIVIDIENDO DATOS CON EVALUACIÓN FINAL RESERVADA...")
        print("⚠️ IMPORTANTE: Los datos de TEST se reservan completamente")
        
        # Usar información de video para división estratificada
        unique_videos = self.video_info[['video_id', 'clase']].copy()
        
        print(f"\n📊 Distribución original:")
        for clase in unique_videos['clase'].unique():
            count = len(unique_videos[unique_videos['clase'] == clase])
            print(f"   {clase}: {count} videos")
        
        # División estratificada con manejo de errores
        try:
            split_info = self._perform_stratified_split(unique_videos, test_size, val_size, random_state)
        except Exception as e:
            print(f"⚠️ División estratificada falló: {e}")
            split_info = self._perform_manual_split(unique_videos, test_size, val_size, random_state)
        
        # Mostrar estadísticas finales
        self._print_split_statistics(split_info)
        
        return split_info
    
    def _perform_stratified_split(self, unique_videos, test_size, val_size, random_state):
        """Realiza división estratificada."""
        train_val_videos, test_videos = train_test_split(
            unique_videos['video_id'].values,
            test_size=test_size,
            stratify=unique_videos['clase'].values,
            random_state=random_state
        )
        
        # Segunda división para validación
        train_val_info = unique_videos[unique_videos['video_id'].isin(train_val_videos)]
        val_size_adjusted = val_size / (1 - test_size)
        
        train_videos, val_videos = train_test_split(
            train_val_videos,
            test_size=val_size_adjusted,
            stratify=train_val_info['clase'].values,
            random_state=random_state
        )
        
        return self._create_split_info(unique_videos, train_videos, val_videos, test_videos)
    
    def _perform_manual_split(self, unique_videos, test_size, val_size, random_state):
        """Realiza división manual cuando la estratificada falla."""
        print("Usando división manual...")
        
        np.random.seed(random_state)
        test_videos = []
        train_val_videos = []
        
        for clase in unique_videos['clase'].unique():
            clase_videos = unique_videos[unique_videos['clase'] == clase]['video_id'].values
            np.random.shuffle(clase_videos)
            
            n_test = max(1, int(len(clase_videos) * test_size))
            test_videos.extend(clase_videos[:n_test])
            train_val_videos.extend(clase_videos[n_test:])
        
        # División train/val
        train_videos = []
        val_videos = []
        
        for clase in unique_videos['clase'].unique():
            clase_train_val = [v for v in train_val_videos 
                             if unique_videos[unique_videos['video_id'] == v]['clase'].iloc[0] == clase]
            
            if len(clase_train_val) > 1:
                n_val = max(1, int(len(clase_train_val) * val_size / (1 - test_size)))
                val_videos.extend(clase_train_val[:n_val])
                train_videos.extend(clase_train_val[n_val:])
            else:
                train_videos.extend(clase_train_val)
        
        return self._create_split_info(unique_videos, train_videos, val_videos, test_videos)
    
    def _create_split_info(self, unique_videos, train_videos, val_videos, test_videos):
        """Crea DataFrame con información de división."""
        split_info = unique_videos.copy()
        
        split_info['split'] = split_info['video_id'].apply(
            lambda x: 'train' if x in train_videos else 
                     'val' if x in val_videos else 'test'
        )
        
        return split_info
    
    def _print_split_statistics(self, split_info):
        """Imprime estadísticas de la división."""
        print(f"\n📊 DIVISIÓN FINAL:")
        try:
            split_counts = split_info.groupby(['split', 'clase']).size().unstack(fill_value=0)
            print(split_counts)
        except Exception as e:
            print(f"Error mostrando tabla: {e}")
        
        totals = split_info['split'].value_counts()
        for split in ['train', 'val', 'test']:
            if split in totals:
                count = totals[split]
                percentage = (count / len(split_info)) * 100
                print(f"   {split.upper()}: {count} videos ({percentage:.1f}%)")
        
        print(f"\n⚠️ ADVERTENCIA CRÍTICA:")
        print(f"   Los {totals.get('test', 0)} videos de TEST están RESERVADOS")
        print(f"   NO se usan para entrenamiento, validación o selección de características")
    
    def scale_features(self, datasets_dict, split_info):
        """
        Escala características usando solo datos de entrenamiento.
        """
        if not datasets_dict:
            print("⚠️ No hay datasets para escalar")
            return {}
        
        print(f"\n🔧 ESCALANDO CARACTERÍSTICAS...")
        
        scaled_datasets = {}
        
        for dataset_name, dataset in datasets_dict.items():
            if dataset is None or dataset.empty:
                continue
                
            print(f"  Escalando dataset {dataset_name}...")
            
            try:
                scaled_data = self._scale_single_dataset(dataset, dataset_name, split_info)
                scaled_datasets.update(scaled_data)
            except Exception as e:
                print(f"    ❌ Error escalando {dataset_name}: {e}")
        
        return scaled_datasets
    
    def _scale_single_dataset(self, dataset, dataset_name, split_info):
        """Escala un dataset individual."""
        # Identificar columnas a escalar
        exclude_patterns = ['video_id', 'Objeto', 'clase', 'Frame', 'split']
        numeric_cols = dataset.select_dtypes(include=[np.number]).columns
        cols_to_scale = [col for col in numeric_cols 
                       if not any(pattern in col for pattern in exclude_patterns)]
        
        if len(cols_to_scale) == 0:
            print(f"    ⚠️ No hay columnas para escalar en {dataset_name}")
            return {}
        
        # Crear máscaras para división
        train_mask = dataset['video_id'].isin(
            split_info[split_info['split'] == 'train']['video_id']
        )
        val_mask = dataset['video_id'].isin(
            split_info[split_info['split'] == 'val']['video_id']
        )
        test_mask = dataset['video_id'].isin(
            split_info[split_info['split'] == 'test']['video_id']
        )
        
        # Dividir datos
        train_data = dataset[train_mask].copy()
        val_data = dataset[val_mask].copy()
        test_data = dataset[test_mask].copy()
        
        if len(train_data) == 0:
            print(f"    ⚠️ No hay datos de entrenamiento para {dataset_name}")
            return {}
        
        # Crear y ajustar escalador
        scaler = StandardScaler()
        
        # Manejar NaN en datos de entrenamiento
        train_features = train_data[cols_to_scale].fillna(0)
        
        # Verificar que no hay valores infinitos
        train_features = train_features.replace([np.inf, -np.inf], 0)
        
        scaler.fit(train_features)
        
        # Aplicar escalado
        train_data[cols_to_scale] = scaler.transform(train_features)
        
        scaled_data = {f'{dataset_name}_train': train_data}
        
        if len(val_data) > 0:
            val_features = val_data[cols_to_scale].fillna(0).replace([np.inf, -np.inf], 0)
            val_data[cols_to_scale] = scaler.transform(val_features)
            scaled_data[f'{dataset_name}_val'] = val_data
            
        if len(test_data) > 0:
            test_features = test_data[cols_to_scale].fillna(0).replace([np.inf, -np.inf], 0)
            test_data[cols_to_scale] = scaler.transform(test_features)
            scaled_data[f'{dataset_name}_test'] = test_data
        
        # Guardar escalador
        scaler_params = {
            'mean_': scaler.mean_,
            'scale_': scaler.scale_,
            'feature_names': cols_to_scale
        }
        
        scaler_file = self.output_dir / f'{dataset_name}_scaler.npy'
        np.save(scaler_file, scaler_params, allow_pickle=True)
        
        print(f"    ✅ {dataset_name}: {len(cols_to_scale)} características escaladas")
        
        return scaled_data
    
    def save_all_data(self, datasets_dict, scaled_datasets, split_info, sequences=None, sequence_info=None):
        """
        Guarda todos los datos procesados.
        """
        print(f"\n💾 GUARDANDO TODOS LOS DATOS PROCESADOS...")
        
        try:
            # Guardar datasets originales
            for name, dataset in datasets_dict.items():
                if dataset is not None and not dataset.empty:
                    file_path = self.output_dir / f'{name}_features.csv'
                    dataset.to_csv(file_path, index=False)
                    print(f"  ✅ {name}_features.csv guardado")
            
            # Guardar datasets escalados
            for name, dataset in scaled_datasets.items():
                if dataset is not None and not dataset.empty:
                    file_path = self.output_dir / f'{name}_scaled.csv'
                    dataset.to_csv(file_path, index=False)
                    print(f"  ✅ {name}_scaled.csv guardado")
            
            # Guardar información de división
            if split_info is not None:
                split_file = self.output_dir / 'data_split.csv'
                split_info.to_csv(split_file, index=False)
                print(f"  ✅ data_split.csv guardado")
            
            # Guardar información de video
            if self.video_info is not None:
                video_file = self.output_dir / 'video_info.csv'
                self.video_info.to_csv(video_file, index=False)
                print(f"  ✅ video_info.csv guardado")
            
            # Guardar datos LSTM si existen
            if sequences is not None and sequence_info is not None:
                self._save_lstm_data(sequences, sequence_info)
            
            # Guardar metadatos
            self._save_metadata(datasets_dict, split_info)
            
            print(f"\n💾 Todos los datos guardados en: {self.output_dir}")
            
        except Exception as e:
            print(f"❌ Error guardando datos: {e}")
            raise
    
    def _save_lstm_data(self, sequences, sequence_info):
        """Guarda datos específicos para LSTM."""
        lstm_dir = self.output_dir / 'lstm_data'
        lstm_dir.mkdir(exist_ok=True)
        
        # Guardar información de secuencias
        seq_info_file = lstm_dir / 'sequence_info.csv'
        sequence_info.to_csv(seq_info_file, index=False)
        
        # Guardar secuencias individuales
        for video_id, sequence in sequences.items():
            seq_file = lstm_dir / f'{video_id}.npy'
            np.save(seq_file, sequence)
        
        print(f"  ✅ Datos LSTM guardados ({len(sequences)} secuencias)")
    
    def _save_metadata(self, datasets_dict, split_info):
        """Guarda metadatos del procesamiento."""
        metadata = {
            'total_videos': len(self.video_info) if self.video_info is not None else 0,
            'classes': list(self.video_info['clase'].unique()) if self.video_info is not None else [],
            'class_distribution': self.video_info['clase'].value_counts().to_dict() if self.video_info is not None else {},
            'total_records': len(self.combined_data) if self.combined_data is not None else 0,
            'split_distribution': split_info['split'].value_counts().to_dict() if split_info is not None else {},
            'use_original_data': self.use_original_data,
            'use_aggregated_data': self.use_aggregated_data,
            'datasets_created': list(datasets_dict.keys()) if datasets_dict else [],
            'processing_timestamp': pd.Timestamp.now().isoformat()
        }
        
        metadata_file = self.output_dir / 'metadata.json'
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"  ✅ metadata.json guardado")
    
    def process_all(self):
        """
        Ejecuta todo el pipeline de procesamiento mejorado.
        """
        print("🚀 INICIANDO PROCESAMIENTO COMPLETO MEJORADO")
        print("="*60)
        
        try:
            # 1. Cargar datos limpiados
            print("\n📥 Paso 1: Cargando datos limpiados...")
            combined_data, video_info = self.load_cleaned_data()
            
            # 2. Preparar datasets según configuración
            print("\n🔧 Paso 2: Preparando datasets...")
            datasets_dict = {}
            
            # Datos originales
            if self.use_original_data:
                print("\n📊 Procesando datos originales...")
                original_data, original_features = self.process_original_data()
                if original_data is not None:
                    datasets_dict['original'] = original_data
            
            # Datos agregados
            if self.use_aggregated_data:
                print("\n🔧 Procesando datos agregados...")
                video_features, object_features, frame_features = self.create_features_by_level()
                
                if video_features is not None:
                    datasets_dict['video'] = video_features
                if object_features is not None:
                    datasets_dict['object'] = object_features
                if frame_features is not None:
                    datasets_dict['frame'] = frame_features
            
            # 3. Crear secuencias LSTM
            print("\n🕐 Paso 3: Creando secuencias LSTM...")
            sequences, sequence_info = None, None
            if self.use_aggregated_data:
                sequences, sequence_info = self.create_lstm_sequences()
            
            # 4. Dividir datos
            print("\n✂️ Paso 4: Dividiendo datos...")
            split_info = self.split_data_optimized()
            
            # 5. Escalar características
            print("\n🔧 Paso 5: Escalando características...")
            scaled_datasets = self.scale_features(datasets_dict, split_info)
            
            # 6. Guardar todo
            print("\n💾 Paso 6: Guardando datos...")
            self.save_all_data(datasets_dict, scaled_datasets, split_info, sequences, sequence_info)
            
            # Resumen final
            self._print_final_summary(datasets_dict, scaled_datasets, sequences)
            
            print(f"\n🎉 PROCESAMIENTO MEJORADO COMPLETADO!")
            print(f"   Datasets creados: {list(datasets_dict.keys())}")
            print(f"   Datos guardados en: {self.output_dir}")
            
            return {
                'datasets': datasets_dict,
                'scaled_datasets': scaled_datasets,
                'split_info': split_info,
                'sequences': sequences,
                'sequence_info': sequence_info,
                'video_info': video_info,
                'combined_data': combined_data
            }
            
        except Exception as e:
            print(f"❌ Error en procesamiento: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def _print_final_summary(self, datasets_dict, scaled_datasets, sequences):
        """Imprime resumen final del procesamiento."""
        print(f"\n📋 RESUMEN FINAL DEL PROCESAMIENTO:")
        
        # Datasets originales
        print(f"\n🗂️ Datasets originales creados:")
        for name, dataset in datasets_dict.items():
            if dataset is not None:
                print(f"   {name}: {len(dataset)} registros")
        
        # Datasets escalados
        print(f"\n⚖️ Datasets escalados creados:")
        splits = {}
        for name in scaled_datasets.keys():
            dataset_name = name.rsplit('_', 1)[0]
            split_type = name.rsplit('_', 1)[1]
            
            if dataset_name not in splits:
                splits[dataset_name] = {}
            splits[dataset_name][split_type] = len(scaled_datasets[name])
        
        for dataset_name, split_counts in splits.items():
            print(f"   {dataset_name}:")
            for split_type, count in split_counts.items():
                print(f"     {split_type}: {count} registros")
        
        # Secuencias LSTM
        if sequences:
            print(f"\n🕐 Secuencias LSTM creadas: {len(sequences)}")
        
        print(f"\n✅ Procesamiento completado exitosamente")

def main():
    """Función principal para ejecutar el procesador."""
    
    # Configuración del procesador
    processor = EnhancedDataProcessor(
        clean_data_dir='./cleaned_data/',
        output_dir='./processed_data_enhanced/',
        use_original_data=True,
        use_aggregated_data=True
    )
    
    # Ejecutar procesamiento completo
    results = processor.process_all()
    
    if results is not None:
        print(f"\n🎯 PROCESAMIENTO EXITOSO!")
        print(f"   Total de videos procesados: {len(results['video_info'])}")
        print(f"   Datasets disponibles: {list(results['datasets'].keys())}")
        
        # Mostrar algunas estadísticas adicionales
        if 'split_info' in results and results['split_info'] is not None:
            split_counts = results['split_info']['split'].value_counts()
            print(f"   División de datos:")
            for split, count in split_counts.items():
                print(f"     {split}: {count} videos")
    
    else:
        print(f"\n❌ PROCESAMIENTO FALLÓ!")
        
    return results

if __name__ == "__main__":
    results = main()