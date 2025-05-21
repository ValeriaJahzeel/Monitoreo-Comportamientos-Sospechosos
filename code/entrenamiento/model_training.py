"""
Módulo para entrenamiento de modelos de clasificación de comportamientos.
Este script entrena modelos MLP, Random Forest y LSTM para clasificar comportamientos
en videos como normales, merodeo o forcejeo.
"""
import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

class ModelTrainer:
    def __init__(self, data_dir='./processed_data/', output_dir='./models/'):
        """
        Inicializa el entrenador de modelos.
        
        Parámetros:
        -----------
        data_dir : str
            Directorio donde se encuentran los datos procesados
        output_dir : str
            Directorio donde se guardarán los modelos entrenados
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.feature_dir = os.path.join(data_dir, 'feature_analysis')
        
        # Crear directorios de salida
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'mlp'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'rf'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'lstm'), exist_ok=True)
        
        # Diccionario para almacenar modelos entrenados
        self.models = {}
        
        # Cargar información sobre conjuntos de datos
        try:
            self.split_info = pd.read_csv(os.path.join(data_dir, 'data_split.csv'))
        except FileNotFoundError:
            print("Error: No se encontró el archivo de división de datos.")
            print("Ejecute primero data_processing.py para generar los archivos necesarios.")
    
    def load_optimal_features(self, level='video'):
        """
        Carga el conjunto óptimo de características para un nivel de agregación.
        
        Parámetros:
        -----------
        level : str
            Nivel de agregación ('video', 'object' o 'frame')
        """
        try:
            # Intentar cargar desde archivo guardado por feature_engineering
            feature_file = os.path.join(self.feature_dir, f'{level}_combined_features.csv')
            if os.path.exists(feature_file):
                feature_df = pd.read_csv(feature_file)
                # Filtrar características con puntuación alta (ocurrencias >= 2)
                features = feature_df[feature_df['count'] >= 2]['feature'].tolist()
                if not features:  # Si no hay suficientes características con count >= 2
                    features = feature_df['feature'].head(15).tolist()
            else:
                # Características predeterminadas si no existe el archivo
                if level == 'video':
                    features = [
                        'Desplazamiento_mean', 'Velocidad_max', 'Aceleracion_std',
                        'Zigzag_mean', 'Area_Trayectoria_max', 'En_Interaccion_mean',
                        'Linealidad_mean', 'Circularidad_mean', 'Densidad_mean',
                        'Es_Ciclico_mean', 'Frame_max', 'num_objetos'
                    ]
                elif level == 'object':
                    features = [
                        'Desplazamiento_sum', 'Velocidad_max', 'Aceleracion_std',
                        'Zigzag_mean', 'Area_Trayectoria_max', 'En_Interaccion_mean',
                        'Linealidad_mean', 'Circularidad_mean', 'Es_Ciclico_mean'
                    ]
                else:  # frame
                    features = [
                        'Desplazamiento_mean', 'Desplazamiento_max', 'Velocidad_mean',
                        'Velocidad_max', 'Aceleracion_mean', 'Densidad_mean',
                        'Linealidad_mean', 'Zigzag_mean', 'Es_Ciclico_mean',
                        'Area_Trayectoria_mean', 'En_Interaccion_mean'
                    ]
            
            print(f"Características óptimas cargadas para nivel '{level}': {len(features)} características")
            return features
            
        except Exception as e:
            print(f"Error al cargar características óptimas: {str(e)}")
            return []
    
    def prepare_data(self, level='video'):
        """
        Prepara los datos para entrenamiento y evaluación de modelos.
        
        Parámetros:
        -----------
        level : str
            Nivel de agregación ('video', 'object' o 'frame')
        """
        print(f"Preparando datos para nivel '{level}'...")
        
        try:
            # Cargar datos escalados
            train_data = pd.read_csv(os.path.join(self.data_dir, f'{level}_train_scaled.csv'))
            val_data = pd.read_csv(os.path.join(self.data_dir, f'{level}_val_scaled.csv'))
            test_data = pd.read_csv(os.path.join(self.data_dir, f'{level}_test_scaled.csv'))
            
            # Cargar características óptimas
            features = self.load_optimal_features(level)
            
            # Preparar conjuntos de entrenamiento, validación y prueba
            X_train = train_data[features]
            y_train = train_data['clase_encoded']
            
            X_val = val_data[features]
            y_val = val_data['clase_encoded']
            
            X_test = test_data[features]
            y_test = test_data['clase_encoded']
            
            data = {
                'X_train': X_train, 'y_train': y_train,
                'X_val': X_val, 'y_val': y_val,
                'X_test': X_test, 'y_test': y_test,
                'features': features
            }
            
            # Guardar nombres de características
            feature_names_path = os.path.join(self.output_dir, f'{level}_features.pkl')
            with open(feature_names_path, 'wb') as f:
                pickle.dump(features, f)
            
            print(f"Datos preparados para nivel '{level}':")
            print(f"  Entrenamiento: {X_train.shape[0]} muestras")
            print(f"  Validación: {X_val.shape[0]} muestras")
            print(f"  Prueba: {X_test.shape[0]} muestras")
            print(f"  Características: {X_train.shape[1]}")
            
            return data
            
        except Exception as e:
            print(f"Error al preparar datos: {str(e)}")
            return None
    
    def train_mlp(self, level='video'):
        """
        Entrena un modelo MLP (Perceptrón Multicapa) para clasificación.
        
        Parámetros:
        -----------
        level : str
            Nivel de agregación ('video', 'object' o 'frame')
        """
        print(f"\nEntrenando modelo MLP para nivel '{level}'...")
        
        # Preparar datos
        data = self.prepare_data(level)
        if not data:
            return None
        
        X_train, y_train = data['X_train'], data['y_train']
        X_val, y_val = data['X_val'], data['y_val']
        
        # Configurar parámetros para grid search
        param_grid = {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive'],
            'max_iter': [300]
        }
        
        # Crear modelo base
        mlp = MLPClassifier(random_state=42)
        
        # Realizar búsqueda de hiperparámetros
        print("Realizando búsqueda de hiperparámetros...")
        grid_search = GridSearchCV(
            mlp, param_grid, cv=3, n_jobs=-1, scoring='f1_weighted', verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        # Obtener mejores parámetros
        best_params = grid_search.best_params_
        print(f"Mejores parámetros: {best_params}")
        
        # Entrenar modelo con mejores parámetros
        best_mlp = MLPClassifier(random_state=42, **best_params)
        best_mlp.fit(X_train, y_train)
        
        # Evaluar en conjunto de validación
        y_val_pred = best_mlp.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        val_f1 = f1_score(y_val, y_val_pred, average='weighted')
        
        print(f"Rendimiento en validación:")
        print(f"  Exactitud: {val_accuracy:.4f}")
        print(f"  F1-Score: {val_f1:.4f}")
        
        # Guardar modelo
        model_path = os.path.join(self.output_dir, 'mlp', f'{level}_mlp.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(best_mlp, f)
        
        # Guardar resultados de la búsqueda
        results_path = os.path.join(self.output_dir, 'mlp', f'{level}_mlp_grid_results.pkl')
        with open(results_path, 'wb') as f:
            pickle.dump(grid_search.cv_results_, f)
        
        # Almacenar modelo en diccionario
        self.models[f'mlp_{level}'] = {
            'model': best_mlp,
            'params': best_params,
            'val_accuracy': val_accuracy,
            'val_f1': val_f1
        }
        
        return best_mlp
    
    def train_random_forest(self, level='video'):
        """
        Entrena un modelo Random Forest para clasificación.
        
        Parámetros:
        -----------
        level : str
            Nivel de agregación ('video', 'object' o 'frame')
        """
        print(f"\nEntrenando modelo Random Forest para nivel '{level}'...")
        
        # Preparar datos
        data = self.prepare_data(level)
        if not data:
            return None
        
        X_train, y_train = data['X_train'], data['y_train']
        X_val, y_val = data['X_val'], data['y_val']
        
        # Configurar parámetros para grid search
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
        
        # Crear modelo base
        rf = RandomForestClassifier(random_state=42)
        
        # Realizar búsqueda de hiperparámetros
        print("Realizando búsqueda de hiperparámetros...")
        grid_search = GridSearchCV(
            rf, param_grid, cv=3, n_jobs=-1, scoring='f1_weighted', verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        # Obtener mejores parámetros
        best_params = grid_search.best_params_
        print(f"Mejores parámetros: {best_params}")
        
        # Entrenar modelo con mejores parámetros
        best_rf = RandomForestClassifier(random_state=42, **best_params)
        best_rf.fit(X_train, y_train)
        
        # Evaluar en conjunto de validación
        y_val_pred = best_rf.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        val_f1 = f1_score(y_val, y_val_pred, average='weighted')
        
        print(f"Rendimiento en validación:")
        print(f"  Exactitud: {val_accuracy:.4f}")
        print(f"  F1-Score: {val_f1:.4f}")
        
        # Guardar modelo
        model_path = os.path.join(self.output_dir, 'rf', f'{level}_rf.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(best_rf, f)
        
        # Guardar resultados de la búsqueda
        results_path = os.path.join(self.output_dir, 'rf', f'{level}_rf_grid_results.pkl')
        with open(results_path, 'wb') as f:
            pickle.dump(grid_search.cv_results_, f)
        
        # Almacenar modelo en diccionario
        self.models[f'rf_{level}'] = {
            'model': best_rf,
            'params': best_params,
            'val_accuracy': val_accuracy,
            'val_f1': val_f1
        }
        
        return best_rf
    
    def prepare_lstm_data(self):
        """
        Prepara datos secuenciales para entrenamiento de LSTM.
        """
        print("Preparando datos secuenciales para LSTM...")
        
        try:
            # Cargar información de secuencias
            seq_info = pd.read_csv(os.path.join(self.data_dir, 'lstm_data', 'sequence_info.csv'))
            
            # Fusionar con información de división
            seq_info = pd.merge(seq_info, self.split_info[['video_id', 'split']], on='video_id')
            
            # Dividir por split
            train_info = seq_info[seq_info['split'] == 'train']
            val_info = seq_info[seq_info['split'] == 'val']
            test_info = seq_info[seq_info['split'] == 'test']
            
            # Función para cargar secuencias
            def load_sequences(info_df, max_seq_len=None):
                sequences = []
                labels = []
                video_ids = []
                
                for _, row in info_df.iterrows():
                    video_id = row['video_id']
                    seq_file = os.path.join(self.data_dir, 'lstm_data', f'{video_id}.npy')
                    
                    if os.path.exists(seq_file):
                        seq = np.load(seq_file)
                        
                        # Normalizar longitud de secuencia si es necesario
                        if max_seq_len:
                            if len(seq) > max_seq_len:
                                # Truncar
                                seq = seq[:max_seq_len]
                            elif len(seq) < max_seq_len:
                                # Rellenar con ceros
                                padding = np.zeros((max_seq_len - len(seq), seq.shape[1]))
                                seq = np.vstack([seq, padding])
                        
                        sequences.append(seq)
                        labels.append(row['clase_encoded'])
                        video_ids.append(video_id)
                
                return sequences, labels, video_ids
            
            # Determinar longitud máxima de secuencia o truncar a un valor razonable
            max_seq_length = min(100, train_info['sequence_length'].median())
            print(f"Usando secuencias de longitud máxima: {max_seq_length}")
            
            # Cargar secuencias
            train_seq, train_labels, train_vids = load_sequences(train_info, max_seq_length)
            val_seq, val_labels, val_vids = load_sequences(val_info, max_seq_length)
            test_seq, test_labels, test_vids = load_sequences(test_info, max_seq_length)
            
            # Convertir a arrays numpy
            X_train = np.array(train_seq)
            y_train = np.array(train_labels)
            X_val = np.array(val_seq)
            y_val = np.array(val_labels)
            X_test = np.array(test_seq)
            y_test = np.array(test_labels)
            
            # Convertir etiquetas a one-hot
            num_classes = len(np.unique(np.concatenate([y_train, y_val, y_test])))
            y_train_onehot = to_categorical(y_train, num_classes)
            y_val_onehot = to_categorical(y_val, num_classes)
            y_test_onehot = to_categorical(y_test, num_classes)
            
            print(f"Datos LSTM preparados:")
            print(f"  Entrenamiento: {len(X_train)} secuencias")
            print(f"  Validación: {len(X_val)} secuencias")
            print(f"  Prueba: {len(X_test)} secuencias")
            print(f"  Forma de secuencias: {X_train[0].shape}")
            
            lstm_data = {
                'X_train': X_train, 'y_train': y_train, 'y_train_onehot': y_train_onehot,
                'X_val': X_val, 'y_val': y_val, 'y_val_onehot': y_val_onehot,
                'X_test': X_test, 'y_test': y_test, 'y_test_onehot': y_test_onehot,
                'max_seq_length': max_seq_length,
                'num_features': X_train[0].shape[1],
                'num_classes': num_classes,
                'train_vids': train_vids,
                'val_vids': val_vids,
                'test_vids': test_vids
            }
            
            return lstm_data
            
        except Exception as e:
            print(f"Error al preparar datos LSTM: {str(e)}")
            return None
    
    def train_lstm(self):
        """
        Entrena un modelo LSTM para clasificación de secuencias temporales.
        """
        print("\nEntrenando modelo LSTM...")
        
        # Preparar datos
        data = self.prepare_lstm_data()
        if not data:
            return None
        
        X_train = data['X_train']
        y_train_onehot = data['y_train_onehot']
        X_val = data['X_val']
        y_val_onehot = data['y_val_onehot']
        
        num_features = data['num_features']
        num_classes = data['num_classes']
        max_seq_length = data['max_seq_length']
        
        # Crear modelo LSTM
        model = Sequential()
        
        # Capa LSTM principal
        model.add(LSTM(units=64, return_sequences=True, 
                      input_shape=(max_seq_length, num_features)))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        
        # Segunda capa LSTM
        model.add(LSTM(units=32))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        
        # Capas densas
        model.add(Dense(units=16, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(units=num_classes, activation='softmax'))
        
        # Compilar modelo
        model.compile(loss='categorical_crossentropy', 
                     optimizer='adam', 
                     metrics=['accuracy'])
        
        # Callbacks para entrenamiento
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint(
                filepath=os.path.join(self.output_dir, 'lstm', 'best_model.h5'),
                monitor='val_accuracy',
                save_best_only=True
            )
        ]
        
        # Entrenar modelo
        print("Entrenando modelo LSTM...")
        history = model.fit(
            X_train, y_train_onehot,
            epochs=100,
            batch_size=16,
            validation_data=(X_val, y_val_onehot),
            callbacks=callbacks,
            verbose=2
        )
        
        # Evaluar en conjunto de validación
        val_loss, val_accuracy = model.evaluate(X_val, y_val_onehot, verbose=0)
        
        print(f"Rendimiento en validación:")
        print(f"  Exactitud: {val_accuracy:.4f}")
        print(f"  Pérdida: {val_loss:.4f}")
        
        # Guardar modelo completo
        model.save(os.path.join(self.output_dir, 'lstm', 'final_model.h5'))
        
        # Guardar historia de entrenamiento
        hist_df = pd.DataFrame(history.history)
        hist_df.to_csv(os.path.join(self.output_dir, 'lstm', 'training_history.csv'), index=False)
        
        # Visualizar historia de entrenamiento
        plt.figure(figsize=(12, 5))
        
        # Gráfico de pérdida
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Entrenamiento')
        plt.plot(history.history['val_loss'], label='Validación')
        plt.title('Pérdida')
        plt.xlabel('Época')
        plt.ylabel('Pérdida')
        plt.legend()
        
        # Gráfico de exactitud
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Entrenamiento')
        plt.plot(history.history['val_accuracy'], label='Validación')
        plt.title('Exactitud')
        plt.xlabel('Época')
        plt.ylabel('Exactitud')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'lstm', 'training_history.png'))
        plt.close()
        
        # Almacenar modelo en diccionario
        self.models['lstm'] = {
            'model': model,
            'val_accuracy': val_accuracy,
            'val_loss': val_loss,
            'data': data
        }
        
        return model
    
    def train_all_models(self):
        """
        Entrena todos los modelos para todos los niveles de agregación.
        """
        results = {}
        
        # Entrenar modelos MLP y Random Forest para cada nivel
        for level in ['video', 'object', 'frame']:
            print(f"\n{'='*50}")
            print(f"ENTRENANDO MODELOS PARA NIVEL '{level.upper()}'")
            print(f"{'='*50}")
            
            # Entrenar MLP
            mlp_model = self.train_mlp(level)
            
            # Entrenar Random Forest
            rf_model = self.train_random_forest(level)
            
            results[level] = {
                'mlp': mlp_model,
                'rf': rf_model
            }
        
        # Entrenar LSTM (solo maneja secuencias temporales)
        print(f"\n{'='*50}")
        print(f"ENTRENANDO MODELO LSTM")
        print(f"{'='*50}")
        
        lstm_model = self.train_lstm()
        results['lstm'] = lstm_model
        
        return results

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train_all_models()