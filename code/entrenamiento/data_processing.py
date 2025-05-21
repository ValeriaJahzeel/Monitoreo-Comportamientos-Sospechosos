"""
Módulo para el procesamiento de datos de comportamientos.
Este script prepara los datos de los CSV de características para el entrenamiento
y prueba de modelos de clasificación.
"""
import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

class DataProcessor:
    def __init__(self, data_dir='./informacion/csv/', output_dir='./processed_data/'):
        """
        Inicializa el procesador de datos.
        
        Parámetros:
        -----------
        data_dir : str
            Directorio donde se encuentran los CSV de características
        output_dir : str
            Directorio donde se guardarán los datos procesados
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.classes = ['normal', 'merodeo', 'forcejeo']
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Crear directorio de salida si no existe
        os.makedirs(output_dir, exist_ok=True)
        
    def load_data(self):
        """
        Carga los datos de todos los CSV disponibles en las carpetas de clases.
        """
        print("Cargando datos...")
        all_data = []
        
        for class_name in self.classes:
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Advertencia: La carpeta {class_dir} no existe. Saltando...")
                continue
                
            csv_files = glob.glob(os.path.join(class_dir, "*.csv"))
            print(f"Encontrados {len(csv_files)} archivos CSV para la clase '{class_name}'")
            
            for csv_file in csv_files:
                video_name = os.path.basename(csv_file).split('.')[0]
                try:
                    # Cargar CSV
                    df = pd.read_csv(csv_file)
                    
                    # Añadir información de clase y video
                    df['clase'] = class_name
                    df['video_id'] = video_name
                    
                    all_data.append(df)
                    print(f"  ✓ Cargado {video_name}: {len(df)} registros")
                except Exception as e:
                    print(f"  ✗ Error al cargar {csv_file}: {str(e)}")
        
        if not all_data:
            raise ValueError("No se pudieron cargar datos de ningún archivo CSV")
            
        # Combinar todos los DataFrames
        self.data = pd.concat(all_data, ignore_index=True)
        print(f"Datos combinados: {len(self.data)} registros totales")
        
        return self.data
    
    def clean_data(self):
        """
        Limpia y preprocesa los datos cargados.
        """
        print("Limpiando datos...")
        
        # Verificar si hay datos cargados
        if not hasattr(self, 'data') or self.data is None:
            raise ValueError("No hay datos cargados. Ejecute load_data() primero.")
        
        # Eliminar duplicados
        self.data = self.data.drop_duplicates()
        
        # Reemplazar valores infinitos o NaN
        self.data.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Imputar valores faltantes
        numeric_cols = self.data.select_dtypes(include=['float64', 'int64']).columns
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        
        for col in numeric_cols:
            self.data[col].fillna(self.data[col].median(), inplace=True)
            
        for col in categorical_cols:
            if col not in ['clase', 'video_id', 'Objeto']:  # No imputar estas columnas
                if not self.data[col].mode().empty:
                    self.data[col].fillna(self.data[col].mode()[0], inplace=True)
                else:
                    self.data[col].fillna("Desconocido", inplace=True)
                    
        print(f"Datos después de limpieza: {len(self.data)} registros")
        return self.data
    
    def prepare_features_by_frame(self):
        """
        Prepara características agregadas por frame para cada video.
        """
        print("Preparando características agregadas por frame...")
        
        # Agrupar por video, frame y calcular estadísticas
        frame_features = self.data.groupby(['video_id', 'Frame']).agg({
            'Desplazamiento': ['mean', 'max', 'std'],
            'Velocidad': ['mean', 'max', 'std'],
            'Aceleracion': ['mean', 'max', 'std'],
            'Densidad': 'mean',
            'Linealidad': 'mean',
            'Circularidad': 'mean',
            'Zigzag': 'mean',
            'Es_Ciclico': 'mean',
            'Area_Trayectoria': 'mean',
            'En_Interaccion': 'mean',
            'clase': 'first'  # Mantener la etiqueta de clase
        }).reset_index()
        
        # Aplanar las columnas multiíndice
        frame_features.columns = ['_'.join(col).strip('_') for col in frame_features.columns.values]
        
        # Codificar la variable de clase
        frame_features['clase_encoded'] = self.label_encoder.fit_transform(frame_features['clase_first'])
        
        # Guardar datos por frame
        frame_features.to_csv(os.path.join(self.output_dir, 'frame_features.csv'), index=False)
        print(f"Características por frame guardadas: {len(frame_features)} registros")
        
        return frame_features
    
    def prepare_features_by_object(self):
        """
        Prepara características agregadas por objeto para cada video.
        """
        print("Preparando características agregadas por objeto...")
        
        # Agrupar por video, objeto y calcular estadísticas temporales
        object_features = self.data.groupby(['video_id', 'Objeto']).agg({
            'Desplazamiento': ['mean', 'max', 'std', 'sum'],
            'Velocidad': ['mean', 'max', 'std'],
            'Aceleracion': ['mean', 'max', 'std'],
            'Linealidad': 'mean',
            'Circularidad': 'mean',
            'Zigzag': 'mean',
            'Es_Ciclico': 'mean',
            'Area_Trayectoria': 'max',
            'En_Interaccion': 'mean',
            'clase': 'first'  # Mantener la etiqueta de clase
        }).reset_index()
        
        # Aplanar las columnas multiíndice
        object_features.columns = ['_'.join(col).strip('_') for col in object_features.columns.values]
        
        # Codificar la variable de clase
        object_features['clase_encoded'] = self.label_encoder.fit_transform(object_features['clase_first'])
        
        # Guardar datos por objeto
        object_features.to_csv(os.path.join(self.output_dir, 'object_features.csv'), index=False)
        print(f"Características por objeto guardadas: {len(object_features)} registros")
        
        return object_features
    
    def prepare_features_by_video(self):
        """
        Prepara características agregadas por video.
        """
        print("Preparando características agregadas por video...")
        
        # Agrupar por video y calcular estadísticas globales
        video_features = self.data.groupby(['video_id']).agg({
            'Frame': 'max',  # Duración aproximada del video
            'Desplazamiento': ['mean', 'max', 'std', 'sum'],
            'Velocidad': ['mean', 'max', 'std'],
            'Aceleracion': ['mean', 'max', 'std'],
            'Densidad': 'mean',
            'Linealidad': 'mean',
            'Circularidad': 'mean',
            'Zigzag': 'mean',
            'Es_Ciclico': 'mean',
            'Area_Trayectoria': ['mean', 'max'],
            'En_Interaccion': 'mean',
            'clase': 'first'  # Mantener la etiqueta de clase
        }).reset_index()
        
        # Aplanar las columnas multiíndice
        video_features.columns = ['_'.join(col).strip('_') for col in video_features.columns.values]
        
        # Codificar la variable de clase
        video_features['clase_encoded'] = self.label_encoder.fit_transform(video_features['clase_first'])
        
        # Obtener conteos de objetos y trayectorias por video
        objetos_por_video = self.data.groupby('video_id')['Objeto'].nunique().reset_index()
        objetos_por_video.columns = ['video_id', 'num_objetos']
        
        # Combinar con las características de video
        video_features = pd.merge(video_features, objetos_por_video, on='video_id', how='left')
        
        # Calcular características adicionales por video
        patron_stats = self.data.groupby(['video_id', 'Patron_Movimiento']).size().unstack(fill_value=0)
        patron_stats = patron_stats.div(patron_stats.sum(axis=1), axis=0)  # Normalizar a proporciones
        
        # Renombrar columnas y combinar con características de video
        patron_stats.columns = [f'patron_{col.lower()}_ratio' for col in patron_stats.columns]
        patron_stats.reset_index(inplace=True)
        
        video_features = pd.merge(video_features, patron_stats, on='video_id', how='left')
        
        # Guardar datos por video
        video_features.to_csv(os.path.join(self.output_dir, 'video_features.csv'), index=False)
        print(f"Características por video guardadas: {len(video_features)} registros")
        
        return video_features
    
    def prepare_features_for_lstm(self):
        """
        Prepara secuencias temporales para entrenamiento de LSTM.
        Agrega características por frame para cada video, manteniendo el orden temporal.
        """
        print("Preparando secuencias temporales para LSTM...")
        
        # Ordenar datos por video y frame
        sorted_data = self.data.sort_values(['video_id', 'Frame'])
        
        # Crear características secuenciales por frame para cada video
        video_sequences = {}
        classes = {}
        
        for video_id, video_data in sorted_data.groupby('video_id'):
            # Agrupar por frame para obtener características agregadas
            frame_data = video_data.groupby('Frame').agg({
                'Desplazamiento': ['mean', 'max'],
                'Velocidad': ['mean', 'max'],
                'Aceleracion': ['mean', 'max'],
                'Densidad': 'mean',
                'Linealidad': 'mean',
                'Circularidad': 'mean',
                'Zigzag': 'mean',
                'Es_Ciclico': 'mean',
                'Area_Trayectoria': 'mean',
                'En_Interaccion': 'mean',
                'clase': 'first'
            })
            
            # Aplanar las columnas multiíndice
            frame_data.columns = ['_'.join(col).strip('_') for col in frame_data.columns.values]
            
            # Extraer características numéricas para la secuencia
            features_cols = [col for col in frame_data.columns if col != 'clase_first']
            sequence = frame_data[features_cols].values
            
            # Almacenar secuencia y clase
            video_sequences[video_id] = sequence
            classes[video_id] = frame_data['clase_first'].iloc[0]
        
        # Guardar datos para LSTM en formato numpy
        os.makedirs(os.path.join(self.output_dir, 'lstm_data'), exist_ok=True)
        
        # Crear un CSV con información de las secuencias y clases
        sequence_info = pd.DataFrame({
            'video_id': list(video_sequences.keys()),
            'sequence_length': [len(seq) for seq in video_sequences.values()],
            'clase': [classes[vid] for vid in video_sequences.keys()]
        })
        
        sequence_info['clase_encoded'] = self.label_encoder.fit_transform(sequence_info['clase'])
        sequence_info.to_csv(os.path.join(self.output_dir, 'lstm_data', 'sequence_info.csv'), index=False)
        
        # Guardar cada secuencia como un archivo numpy
        for video_id, sequence in video_sequences.items():
            np.save(os.path.join(self.output_dir, 'lstm_data', f'{video_id}.npy'), sequence)
        
        print(f"Secuencias temporales guardadas para {len(video_sequences)} videos")
        return sequence_info
    
    def split_data(self, test_size=0.2, val_size=0.1, random_state=42):
        """
        Divide los datos procesados en conjuntos de entrenamiento, validación y prueba.
        
        Parámetros:
        -----------
        test_size : float
            Proporción de datos para el conjunto de prueba
        val_size : float
            Proporción de datos para el conjunto de validación
        random_state : int
            Semilla para reproducibilidad
        """
        print("Dividiendo datos en conjuntos de entrenamiento, validación y prueba...")
        
        # Obtener conjuntos de IDs de videos únicos
        video_ids = self.data['video_id'].unique()
        
        # Primera división: separar conjunto de prueba
        train_val_ids, test_ids = train_test_split(
            video_ids, test_size=test_size, random_state=random_state, 
            stratify=self.data.drop_duplicates('video_id')['clase']
        )
        
        # Segunda división: separar conjunto de validación de entrenamiento
        val_size_adjusted = val_size / (1 - test_size)  # Ajustar proporción
        train_ids, val_ids = train_test_split(
            train_val_ids, test_size=val_size_adjusted, random_state=random_state,
            stratify=self.data[self.data['video_id'].isin(train_val_ids)].drop_duplicates('video_id')['clase']
        )
        
        # Guardar los IDs de cada conjunto
        splits = {
            'train': train_ids,
            'val': val_ids,
            'test': test_ids
        }
        
        # Crear DataFrame con la información de división
        split_info = pd.DataFrame({
            'video_id': video_ids,
            'split': ['train' if vid in train_ids else 'val' if vid in val_ids else 'test' for vid in video_ids]
        })
        
        # Añadir información de clase
        video_classes = self.data.drop_duplicates('video_id')[['video_id', 'clase']]
        split_info = pd.merge(split_info, video_classes, on='video_id')
        
        # Guardar información de división
        split_info.to_csv(os.path.join(self.output_dir, 'data_split.csv'), index=False)
        
        print(f"División completada: {len(train_ids)} videos de entrenamiento, "
              f"{len(val_ids)} de validación, {len(test_ids)} de prueba")
        
        return splits
    
    def scale_features(self, frame_features, object_features, video_features):
        """
        Escala las características numéricas usando StandardScaler.
        
        Parámetros:
        -----------
        frame_features : DataFrame
            Características agregadas por frame
        object_features : DataFrame
            Características agregadas por objeto
        video_features : DataFrame
            Características agregadas por video
        """
        print("Escalando características...")
        
        # Cargar información de división
        split_info = pd.read_csv(os.path.join(self.output_dir, 'data_split.csv'))
        
        datasets = {
            'frame': frame_features,
            'object': object_features,
            'video': video_features
        }
        
        scaled_datasets = {}
        
        for name, dataset in datasets.items():
            # Identificar qué características escalar (excluyendo identificadores y clases)
            id_cols = ['video_id']
            if name == 'frame':
                id_cols.append('Frame_')
            elif name == 'object':
                id_cols.append('Objeto_')
                
            target_cols = ['clase_first', 'clase_encoded']
            
            # Identificar columnas numéricas a escalar
            cols_to_scale = dataset.select_dtypes(include=['float64', 'int64']).columns
            cols_to_scale = [col for col in cols_to_scale if not any(col.startswith(id_col) for id_col in id_cols) 
                            and col not in target_cols]
            
            # Dividir en conjuntos de entrenamiento, validación y prueba
            train_data = dataset[dataset['video_id'].isin(split_info[split_info['split'] == 'train']['video_id'])]
            val_data = dataset[dataset['video_id'].isin(split_info[split_info['split'] == 'val']['video_id'])]
            test_data = dataset[dataset['video_id'].isin(split_info[split_info['split'] == 'test']['video_id'])]
            
            # Ajustar scaler solo con datos de entrenamiento
            scaler = StandardScaler()
            scaler.fit(train_data[cols_to_scale])
            
            # Aplicar transformación a todos los conjuntos
            for split_name, split_data in zip(['train', 'val', 'test'], [train_data, val_data, test_data]):
                # Crear copia para no modificar el original
                scaled_data = split_data.copy()
                
                # Aplicar transformación
                if len(split_data) > 0:
                    scaled_data[cols_to_scale] = scaler.transform(split_data[cols_to_scale])
                
                # Guardar datos escalados
                scaled_data.to_csv(os.path.join(self.output_dir, f'{name}_{split_name}_scaled.csv'), index=False)
                
                # Almacenar para retornar
                key = f'{name}_{split_name}'
                scaled_datasets[key] = scaled_data
            
            # Guardar scaler para futuras transformaciones
            np.save(os.path.join(self.output_dir, f'{name}_scaler_mean.npy'), scaler.mean_)
            np.save(os.path.join(self.output_dir, f'{name}_scaler_scale.npy'), scaler.scale_)
        
        print("Escalado de características completado")
        return scaled_datasets
    
    def process_all(self):
        """
        Ejecuta todos los pasos de procesamiento en secuencia.
        """
        # Cargar datos
        self.load_data()
        
        # Limpiar datos
        self.clean_data()
        
        # Preparar características por nivel
        frame_features = self.prepare_features_by_frame()
        object_features = self.prepare_features_by_object()
        video_features = self.prepare_features_by_video()
        
        # Preparar secuencias para LSTM
        self.prepare_features_for_lstm()
        
        # Dividir datos
        self.split_data()
        
        # Escalar características
        scaled_data = self.scale_features(frame_features, object_features, video_features)
        
        print("Procesamiento de datos completado con éxito")
        return scaled_data

if __name__ == "__main__":
    processor = DataProcessor()
    processor.process_all()