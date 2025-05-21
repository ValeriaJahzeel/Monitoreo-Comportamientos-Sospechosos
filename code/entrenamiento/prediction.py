"""
Módulo para realizar predicciones sobre nuevos videos.
Este script carga los modelos entrenados y realiza predicciones para
clasificar nuevos videos en comportamientos normales, merodeo o forcejeo.
"""
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import argparse
import glob

class BehaviorPredictor:
    def __init__(self, models_dir='./results/models/', data_dir='./results/processed_data/'):
        """
        Inicializa el predictor de comportamientos.
        
        Parámetros:
        -----------
        models_dir : str
            Directorio donde se encuentran los modelos entrenados
        data_dir : str
            Directorio donde se encuentran los datos procesados y escaladores
        """
        self.models_dir = models_dir
        self.data_dir = data_dir
        self.models = {}
        self.scalers = {}
        self.features = {}
        self.class_names = ['normal', 'merodeo', 'forcejeo']
        
        # Cargar modelos
        self.load_models()
    
    def load_models(self):
        """
        Carga todos los modelos entrenados y sus configuraciones.
        """
        print("Cargando modelos entrenados...")
        
        # Cargar modelos MLP y RF para cada nivel
        for level in ['video', 'object', 'frame']:
            # Cargar características seleccionadas
            features_path = os.path.join(self.models_dir, f'{level}_features.pkl')
            if os.path.exists(features_path):
                with open(features_path, 'rb') as f:
                    self.features[level] = pickle.load(f)
                print(f"  ✓ Cargadas características para nivel '{level}'")
            
            # Cargar escaladores
            scaler_mean_path = os.path.join(self.data_dir, f'{level}_scaler_mean.npy')
            scaler_scale_path = os.path.join(self.data_dir, f'{level}_scaler_scale.npy')
            
            if os.path.exists(scaler_mean_path) and os.path.exists(scaler_scale_path):
                scaler = StandardScaler()
                scaler.mean_ = np.load(scaler_mean_path)
                scaler.scale_ = np.load(scaler_scale_path)
                self.scalers[level] = scaler
                print(f"  ✓ Cargado escalador para nivel '{level}'")
            
            # Cargar modelo MLP
            mlp_path = os.path.join(self.models_dir, 'mlp', f'{level}_mlp.pkl')
            if os.path.exists(mlp_path):
                with open(mlp_path, 'rb') as f:
                    self.models[f'mlp_{level}'] = pickle.load(f)
                print(f"  ✓ Cargado modelo MLP para nivel '{level}'")
            
            # Cargar modelo Random Forest
            rf_path = os.path.join(self.models_dir, 'rf', f'{level}_rf.pkl')
            if os.path.exists(rf_path):
                with open(rf_path, 'rb') as f:
                    self.models[f'rf_{level}'] = pickle.load(f)
                print(f"  ✓ Cargado modelo Random Forest para nivel '{level}'")
        
        # Cargar modelo LSTM
        lstm_path = os.path.join(self.models_dir, 'lstm', 'final_model.h5')
        if os.path.exists(lstm_path):
            self.models['lstm'] = load_model(lstm_path)
            print(f"  ✓ Cargado modelo LSTM")
        
        # Verificar si se cargó al menos un modelo
        if not self.models:
            raise ValueError("No se pudo cargar ningún modelo. Verifique que los modelos estén entrenados.")
        
        print(f"Total de modelos cargados: {len(self.models)}")
    
    def preprocess_csv(self, csv_path, level='video'):
        """
        Preprocesa un archivo CSV para realizar predicciones.
        
        Parámetros:
        -----------
        csv_path : str
            Ruta al archivo CSV con características extraídas
        level : str
            Nivel de agregación para preprocesar ('video', 'object', 'frame')
        """
        print(f"Preprocesando archivo CSV para nivel '{level}'...")
        
        try:
            # Cargar CSV
            df = pd.read_csv(csv_path)
            
            # Verificar que existen las columnas necesarias
            required_cols = set(['Frame', 'Objeto', 'Desplazamiento', 'Velocidad'])
            if not required_cols.issubset(df.columns):
                missing = required_cols - set(df.columns)
                raise ValueError(f"El archivo CSV no contiene todas las columnas necesarias. Faltan: {missing}")
            
            # Preparar características según nivel de agregación
            if level == 'video':
                # Agregar por video
                video_id = os.path.basename(csv_path).split('.')[0]
                agg_df = df.groupby(['Objeto']).agg({
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
                    'En_Interaccion': 'mean'
                }).reset_index()
                
                # Aplanar columnas multiíndice
                agg_df.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in agg_df.columns]
                
                # Calcular número de objetos
                agg_df['num_objetos'] = len(agg_df)
                
                # Promediar todas las características por objeto para tener una fila por video
                video_df = agg_df.drop('Objeto', axis=1).mean().to_frame().T
                video_df['video_id'] = video_id
                
                processed_df = video_df
                
            elif level == 'object':
                # Agregar por objeto
                agg_df = df.groupby(['Objeto']).agg({
                    'Desplazamiento': ['mean', 'max', 'std', 'sum'],
                    'Velocidad': ['mean', 'max', 'std'],
                    'Aceleracion': ['mean', 'max', 'std'],
                    'Linealidad': 'mean',
                    'Circularidad': 'mean',
                    'Zigzag': 'mean',
                    'Es_Ciclico': 'mean',
                    'Area_Trayectoria': 'max',
                    'En_Interaccion': 'mean'
                }).reset_index()
                
                # Aplanar columnas multiíndice
                agg_df.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in agg_df.columns]
                
                processed_df = agg_df
                
            elif level == 'frame':
                # Agregar por frame
                agg_df = df.groupby(['Frame']).agg({
                    'Desplazamiento': ['mean', 'max', 'std'],
                    'Velocidad': ['mean', 'max', 'std'],
                    'Aceleracion': ['mean', 'max', 'std'],
                    'Densidad': 'mean',
                    'Linealidad': 'mean',
                    'Circularidad': 'mean',
                    'Zigzag': 'mean',
                    'Es_Ciclico': 'mean',
                    'Area_Trayectoria': 'mean',
                    'En_Interaccion': 'mean'
                }).reset_index()
                
                # Aplanar columnas multiíndice
                agg_df.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in agg_df.columns]
                
                processed_df = agg_df
            
            else:
                raise ValueError(f"Nivel de agregación no válido: {level}")
            
            # Verificar que existen todas las características necesarias
            if level in self.features and self.features[level]:
                required_features = set(self.features[level])
                available_features = set(processed_df.columns)
                
                if not required_features.issubset(available_features):
                    missing = required_features - available_features
                    # Crear columnas faltantes con valores predeterminados
                    for col in missing:
                        processed_df[col] = 0
                    print(f"  Advertencia: Se añadieron columnas faltantes con valores predeterminados: {missing}")
            
            return processed_df
            
        except Exception as e:
            print(f"Error al preprocesar CSV: {str(e)}")
            return None
    
    def scale_features(self, df, level='video'):
        """
        Escala las características usando el escalador entrenado.
        
        Parámetros:
        -----------
        df : DataFrame
            DataFrame con características a escalar
        level : str
            Nivel de agregación ('video', 'object', 'frame')
        """
        if level not in self.scalers:
            print(f"Advertencia: No se encontró escalador para nivel '{level}'. No se aplicará escalado.")
            return df
        
        # Aplicar escalado solo a las características numéricas seleccionadas
        scaled_df = df.copy()
        
        if level in self.features and self.features[level]:
            features = self.features[level]
            # Asegurar que todas las características estén presentes
            for feature in features:
                if feature not in scaled_df.columns:
                    scaled_df[feature] = 0
            
            # Aplicar escalado
            scaled_df[features] = self.scalers[level].transform(scaled_df[features])
        
        return scaled_df
    
    def prepare_lstm_data(self, csv_path, max_seq_length=None):
        """
        Prepara datos secuenciales para predicción con modelo LSTM.
        
        Parámetros:
        -----------
        csv_path : str
            Ruta al archivo CSV con características extraídas
        max_seq_length : int, optional
            Longitud máxima de secuencia
        """
        try:
            # Cargar CSV
            df = pd.read_csv(csv_path)
            
            # Verificar que existen las columnas necesarias
            required_cols = set(['Frame', 'Objeto', 'Desplazamiento', 'Velocidad'])
            if not required_cols.issubset(df.columns):
                missing = required_cols - set(df.columns)
                raise ValueError(f"El archivo CSV no contiene todas las columnas necesarias. Faltan: {missing}")
            
            # Ordenar por frame
            df = df.sort_values('Frame')
            
            # Agregar por frame
            frame_data = df.groupby('Frame').agg({
                'Desplazamiento': ['mean', 'max'],
                'Velocidad': ['mean', 'max'],
                'Aceleracion': ['mean', 'max'],
                'Densidad': 'mean',
                'Linealidad': 'mean',
                'Circularidad': 'mean',
                'Zigzag': 'mean',
                'Es_Ciclico': 'mean',
                'Area_Trayectoria': 'mean',
                'En_Interaccion': 'mean'
            })
            
            # Aplanar las columnas multiíndice
            frame_data.columns = ['_'.join(col).strip('_') for col in frame_data.columns.values]
            
            # Extraer características numéricas para la secuencia
            features_cols = frame_data.columns
            sequence = frame_data[features_cols].values
            
            # Comprobar si se proporcionó una longitud máxima de secuencia
            if max_seq_length is None:
                # Intentar obtener del modelo
                if 'lstm' in self.models:
                    try:
                        # Obtener la forma de entrada esperada
                        input_shape = self.models['lstm'].input_shape
                        if input_shape and len(input_shape) > 1:
                            max_seq_length = input_shape[1]
                    except:
                        max_seq_length = 100  # Valor predeterminado
                else:
                    max_seq_length = 100
            
            # Normalizar longitud de secuencia
            if len(sequence) > max_seq_length:
                # Truncar
                sequence = sequence[:max_seq_length]
            elif len(sequence) < max_seq_length:
                # Rellenar con ceros
                padding = np.zeros((max_seq_length - len(sequence), sequence.shape[1]))
                sequence = np.vstack([sequence, padding])
            
            # Expandir dimensiones para batch_size=1
            sequence = np.expand_dims(sequence, axis=0)
            
            return sequence
            
        except Exception as e:
            print(f"Error al preparar datos LSTM: {str(e)}")
            return None
    
    def predict(self, csv_path, model_type='all', level='video'):
        """
        Realiza predicción de comportamiento para un archivo CSV.
        
        Parámetros:
        -----------
        csv_path : str
            Ruta al archivo CSV con características extraídas
        model_type : str
            Tipo de modelo a usar ('mlp', 'rf', 'lstm', 'all')
        level : str
            Nivel de agregación para modelos no secuenciales ('video', 'object', 'frame')
        """
        print(f"Realizando predicción para {csv_path}...")
        
        results = {}
        
        # Si se solicita predicción para todos los modelos
        if model_type == 'all':
            # Predicción con modelos MLP y RF
            for model_type in ['mlp', 'rf']:
                model_name = f"{model_type}_{level}"
                if model_name in self.models:
                    # Preprocesar datos
                    processed_df = self.preprocess_csv(csv_path, level)
                    if processed_df is None:
                        continue
                    
                    # Escalar características
                    scaled_df = self.scale_features(processed_df, level)
                    
                    # Preparar características
                    if level in self.features and self.features[level]:
                        features = self.features[level]
                        X = scaled_df[features]
                    else:
                        # Usar todas las características numéricas excepto identificadores
                        exclude_cols = ['video_id', 'Frame', 'Objeto']
                        X = scaled_df.select_dtypes(include=['float64', 'int64']).drop(
                            [col for col in exclude_cols if col in scaled_df.columns], axis=1)
                    
                    # Realizar predicción
                    model = self.models[model_name]
                    prediction = model.predict(X)
                    
                    # Si es un solo valor (nivel video), convertir a array
                    if level == 'video' and not isinstance(prediction, np.ndarray):
                        prediction = np.array([prediction])
                    
                    # Obtener etiquetas de clase
                    if hasattr(model, 'classes_'):
                        class_labels = [self.class_names[i] if i < len(self.class_names) else f"Clase {i}" 
                                      for i in prediction]
                    else:
                        class_labels = [self.class_names[i] if i < len(self.class_names) else f"Clase {i}" 
                                      for i in prediction]
                    
                    # Obtener probabilidades si el modelo lo soporta
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(X)
                    else:
                        proba = None
                    
                    results[model_name] = {
                        'predictions': prediction,
                        'class_labels': class_labels,
                        'probabilities': proba
                    }
            
            # Predicción con modelo LSTM
            if 'lstm' in self.models:
                # Preparar datos secuenciales
                sequence = self.prepare_lstm_data(csv_path)
                if sequence is not None:
                    # Realizar predicción
                    model = self.models['lstm']
                    pred_proba = model.predict(sequence)
                    prediction = np.argmax(pred_proba, axis=1)
                    
                    # Obtener etiqueta de clase
                    class_labels = [self.class_names[i] if i < len(self.class_names) else f"Clase {i}" 
                                  for i in prediction]
                    
                    results['lstm'] = {
                        'predictions': prediction,
                        'class_labels': class_labels,
                        'probabilities': pred_proba
                    }
        
        # Si se solicita predicción para un modelo específico
        else:
            model_name = f"{model_type}_{level}" if model_type != 'lstm' else 'lstm'
            
            if model_name not in self.models:
                print(f"Error: No se encontró el modelo '{model_name}'")
                return None
            
            if model_type == 'lstm':
                # Preparar datos secuenciales
                sequence = self.prepare_lstm_data(csv_path)
                if sequence is not None:
                    # Realizar predicción
                    model = self.models[model_name]
                    pred_proba = model.predict(sequence)
                    prediction = np.argmax(pred_proba, axis=1)
                    
                    # Obtener etiqueta de clase
                    class_labels = [self.class_names[i] if i < len(self.class_names) else f"Clase {i}" 
                                  for i in prediction]
                    
                    results[model_name] = {
                        'predictions': prediction,
                        'class_labels': class_labels,
                        'probabilities': pred_proba
                    }
            else:
                # Preprocesar datos
                processed_df = self.preprocess_csv(csv_path, level)
                if processed_df is not None:
                    # Escalar características
                    scaled_df = self.scale_features(processed_df, level)
                    
                    # Preparar características
                    if level in self.features and self.features[level]:
                        features = self.features[level]
                        X = scaled_df[features]
                    else:
                        # Usar todas las características numéricas excepto identificadores
                        exclude_cols = ['video_id', 'Frame', 'Objeto']
                        X = scaled_df.select_dtypes(include=['float64', 'int64']).drop(
                            [col for col in exclude_cols if col in scaled_df.columns], axis=1)
                    
                    # Realizar predicción
                    model = self.models[model_name]
                    prediction = model.predict(X)
                    
                    # Si es un solo valor (nivel video), convertir a array
                    if level == 'video' and not isinstance(prediction, np.ndarray):
                        prediction = np.array([prediction])
                    
                    # Obtener etiquetas de clase
                    if hasattr(model, 'classes_'):
                        class_labels = [self.class_names[i] if i < len(self.class_names) else f"Clase {i}" 
                                       for i in prediction]
                    else:
                        class_labels = [self.class_names[i] if i < len(self.class_names) else f"Clase {i}" 
                                       for i in prediction]
                    
                    # Obtener probabilidades si el modelo lo soporta
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(X)
                    else:
                        proba = None
                    
                    results[model_name] = {
                        'predictions': prediction,
                        'class_labels': class_labels,
                        'probabilities': proba
                    }
        
        return results
    
    def ensemble_prediction(self, results):
        """
        Realiza una predicción combinada usando todos los modelos disponibles.
        
        Parámetros:
        -----------
        results : dict
            Diccionario con resultados de predicción de diferentes modelos
        """
        if not results:
            print("Error: No hay resultados de predicción para combinar")
            return None
        
        # Contar votos para cada clase
        votes = {}
        
        for model_name, model_results in results.items():
            # Solo considerar predicciones de nivel de video
            if 'video' in model_name or model_name == 'lstm':
                class_labels = model_results['class_labels']
                
                # Para modelos LSTM y nivel de video, usar solo la primera predicción
                if len(class_labels) == 1 or model_name == 'lstm':
                    label = class_labels[0]
                    votes[label] = votes.get(label, 0) + 1
        
        # Si no hay votos, devolver None
        if not votes:
            return None
        
        # Encontrar la clase con más votos
        max_votes = max(votes.values())
        top_classes = [cls for cls, vote in votes.items() if vote == max_votes]
        
        # En caso de empate, dar prioridad a modelos específicos
        if len(top_classes) > 1:
            # Priorizar primero LSTM, luego RF, luego MLP
            priority_models = ['lstm', 'rf_video', 'mlp_video']
            for model in priority_models:
                if model in results:
                    label = results[model]['class_labels'][0]
                    if label in top_classes:
                        return label
            
            # Si sigue habiendo empate, devolver cualquiera de las clases con más votos
            return top_classes[0]
        else:
            return top_classes[0]
    
    def visualize_prediction(self, results, output_dir='./predictions'):
        """
        Genera visualizaciones de las predicciones.
        
        Parámetros:
        -----------
        results : dict
            Diccionario con resultados de predicción
        output_dir : str
            Directorio donde guardar las visualizaciones
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Solo visualizar si hay probabilidades disponibles
        models_with_proba = {model: res for model, res in results.items() 
                          if res['probabilities'] is not None}
        
        if not models_with_proba:
            print("No hay probabilidades disponibles para visualizar")
            return
        
        # Crear histogramas de probabilidades
        plt.figure(figsize=(12, 8))
        
        # Agregar al máximo 4 modelos para visualizar
        models_to_plot = list(models_with_proba.keys())[:4]
        
        for i, model_name in enumerate(models_to_plot):
            res = models_with_proba[model_name]
            proba = res['probabilities']
            
            # Para nivel de video o LSTM, usar solo la primera predicción
            if proba.shape[0] == 1 or model_name == 'lstm':
                proba = proba[0]
                
                # Crear subgráfico
                plt.subplot(2, 2, i+1)
                plt.bar(range(len(proba)), proba, color='skyblue')
                plt.xticks(range(len(proba)), 
                          [self.class_names[i] if i < len(self.class_names) else f"Clase {i}" 
                           for i in range(len(proba))])
                plt.title(f"Probabilidades - {model_name}")
                plt.ylabel('Probabilidad')
                plt.ylim(0, 1)
                
                # Marcar la clase predicha
                pred_idx = np.argmax(proba)
                plt.bar(pred_idx, proba[pred_idx], color='red')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'prediction_probabilities.png'))
        plt.close()
        
        # Crear gráfico de barras de confianza por modelo
        ensemble_prediction = self.ensemble_prediction(results)
        
        plt.figure(figsize=(10, 6))
        
        models = []
        confidences = []
        colors = []
        
        for model_name, res in models_with_proba.items():
            # Solo considerar modelos de nivel de video o LSTM
            if 'video' in model_name or model_name == 'lstm':
                proba = res['probabilities']
                pred_idx = np.argmax(proba[0] if proba.shape[0] == 1 else proba)
                confidence = proba[0, pred_idx] if proba.shape[0] == 1 else proba[pred_idx]
                
                models.append(model_name)
                confidences.append(confidence)
                
                # Colorear según si coincide con la predicción final
                if res['class_labels'][0] == ensemble_prediction:
                    colors.append('green')
                else:
                    colors.append('red')
        
        plt.barh(models, confidences, color=colors)
        plt.xlabel('Confianza')
        plt.title(f'Confianza por Modelo - Predicción Final: {ensemble_prediction}')
        plt.xlim(0, 1)
        
        # Añadir valores en las barras
        for i, conf in enumerate(confidences):
            plt.text(conf + 0.01, i, f'{conf:.2f}', va='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'model_confidence.png'))
        plt.close()
    
    def predict_csv_files(self, csv_dir, output_dir='./predictions', model_type='all', level='video'):
        """
        Realiza predicciones para todos los archivos CSV en un directorio.
        
        Parámetros:
        -----------
        csv_dir : str
            Directorio con archivos CSV
        output_dir : str
            Directorio donde guardar los resultados
        model_type : str
            Tipo de modelo a usar ('mlp', 'rf', 'lstm', 'all')
        level : str
            Nivel de agregación para modelos no secuenciales ('video', 'object', 'frame')
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Buscar archivos CSV
        csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
        
        if not csv_files:
            print(f"No se encontraron archivos CSV en {csv_dir}")
            return
        
        print(f"Procesando {len(csv_files)} archivos CSV...")
        
        # Resultados
        all_results = []
        
        for csv_file in csv_files:
            video_id = os.path.basename(csv_file).split('.')[0]
            print(f"Analizando video: {video_id}")
            
            # Realizar predicción
            results = self.predict(csv_file, model_type, level)
            
            if not results:
                print(f"  ✗ No se pudo realizar predicción para {video_id}")
                continue
            
            # Realizar predicción combinada
            ensemble_pred = self.ensemble_prediction(results)
            
            # Visualizar predicción
            viz_dir = os.path.join(output_dir, video_id)
            os.makedirs(viz_dir, exist_ok=True)
            self.visualize_prediction(results, viz_dir)
            
            # Recopilar resultados
            result_entry = {
                'video_id': video_id,
                'ensemble_prediction': ensemble_pred
            }
            
            # Añadir predicciones individuales
            for model_name, res in results.items():
                # Solo modelos de nivel video o LSTM (una predicción por video)
                if 'video' in model_name or model_name == 'lstm':
                    result_entry[f'{model_name}_prediction'] = res['class_labels'][0]
            
            all_results.append(result_entry)
            
            print(f"  ✓ Predicción para {video_id}: {ensemble_pred}")
        
        # Guardar resultados en CSV
        if all_results:
            results_df = pd.DataFrame(all_results)
            results_df.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)
            
            # Generar gráfico de resumen
            plt.figure(figsize=(10, 6))
            
            # Contar ocurrencias de cada clase
            pred_counts = results_df['ensemble_prediction'].value_counts()
            
            plt.pie(pred_counts, labels=pred_counts.index, autopct='%1.1f%%',
                  shadow=True, startangle=90)
            plt.axis('equal')
            plt.title('Distribución de Predicciones')
            
            plt.savefig(os.path.join(output_dir, 'prediction_distribution.png'))
            plt.close()
        
        print(f"Predicciones completadas. Resultados guardados en {output_dir}")
        
        return all_results

if __name__ == "__main__":
    # Configurar parser de argumentos
    parser = argparse.ArgumentParser(description='Predictor de comportamientos sospechosos')
    
    parser.add_argument('--csv_dir', required=True,
                       help='Directorio con archivos CSV de características')
    parser.add_argument('--models_dir', default='./results/models/',
                       help='Directorio con modelos entrenados')
    parser.add_argument('--data_dir', default='./results/processed_data/',
                       help='Directorio con datos procesados')
    parser.add_argument('--output_dir', default='./predictions/',
                       help='Directorio para guardar resultados')
    parser.add_argument('--model_type', default='all', 
                        choices=['mlp', 'rf', 'lstm', 'all'],
                       help='Tipo de modelo a usar')
    parser.add_argument('--level', default='video', 
                        choices=['video', 'object', 'frame'],
                       help='Nivel de agregación para modelos no secuenciales')
    
    args = parser.parse_args()
    
    # Crear predictor
    predictor = BehaviorPredictor(models_dir=args.models_dir, data_dir=args.data_dir)
    
    # Realizar predicciones
    predictor.predict_csv_files(
        args.csv_dir, 
        output_dir=args.output_dir,
        model_type=args.model_type, 
        level=args.level
    )