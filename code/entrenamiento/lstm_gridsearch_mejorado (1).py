"""
Entrenador LSTM simplificado para clasificación de videos usando secuencias temporales.
Extrae solo características básicas y mantiene la información temporal.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import GroupShuffleSplit

from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import warnings
warnings.filterwarnings('ignore')

class SimpleVideoLSTMTrainer:
    """
    Entrenador LSTM simplificado para secuencias de video.
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model = None
        self.sequences = []
        self.labels = []
        self.video_names = []
        self.results = {}
        
        # Configurar TensorFlow para usar menos memoria
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)
        
        # Configurar semilla
        tf.random.set_seed(random_state)
        np.random.seed(random_state)
    
    def extract_sequence_features(self, csv_path, video_class):
        """
        Extrae secuencia temporal con características básicas importantes.
        """
        try:
            df = pd.read_csv(csv_path)
            
            if df.empty:
                return None, None, None
            
            video_name = Path(csv_path).stem
            
            # Características básicas más importantes para LSTM
            feature_cols = [
                'Velocidad', 'Aceleracion', 'Cambio_Direccion', 'Linealidad', 'Circularidad',
                'Zigzag', 'Densidad', 'Area_Trayectoria', 'Centroide_X', 'Centroide_Y'
            ]

            
            # Verificar qué columnas existen
            available_cols = [col for col in feature_cols if col in df.columns]
            
            if len(available_cols) == 0:
                print(f"⚠️ No se encontraron características básicas en {csv_path}")
                return None, None, None
            
            # Extraer características por frame
            sequence_data = []
            
            # Agrupar por frame para mantener orden temporal
            if 'Frame' in df.columns:
                frames = sorted(df['Frame'].unique())
                
                for frame in frames:
                    frame_data = df[df['Frame'] == frame]
                    
                    # Agregar características del frame (promedio si hay múltiples objetos)
                    frame_features = []
                    
                    for col in available_cols:
                        if col in frame_data.columns:
                            values = frame_data[col].dropna()
                            if len(values) > 0:
                                frame_features.append(values.mean())
                            else:
                                frame_features.append(0.0)
                        else:
                            frame_features.append(0.0)
                    
                    # Agregar características adicionales del frame
                    # Número de objetos en el frame
                    if 'Objeto' in frame_data.columns:
                        num_objetos = frame_data['Objeto'].nunique()
                    else:
                        num_objetos = len(frame_data)
                    frame_features.append(num_objetos)
                    
                    # Porcentaje de interacción en el frame
                    if 'En_Interaccion' in frame_data.columns:
                        pct_interaccion = (frame_data['En_Interaccion'] == 1).mean()
                    else:
                        pct_interaccion = 0.0
                    frame_features.append(pct_interaccion)
                    
                    # Indicador de movimiento cíclico
                    if 'Es_Ciclico' in frame_data.columns:
                        hay_ciclico = (frame_data['Es_Ciclico'] == 1).any()
                    else:
                        hay_ciclico = False
                    frame_features.append(float(hay_ciclico))
                    
                    sequence_data.append(frame_features)
                    
            else:
                # Si no hay columna Frame, usar cada fila como un paso temporal
                for _, row in df.iterrows():
                    frame_features = []
                    
                    for col in available_cols:
                        if col in row and pd.notna(row[col]):
                            frame_features.append(float(row[col]))
                        else:
                            frame_features.append(0.0)
                    
                    # Características adicionales
                    frame_features.append(1.0)  # num_objetos = 1
                    
                    if 'En_Interaccion' in row:
                        frame_features.append(float(row['En_Interaccion'] == 1))
                    else:
                        frame_features.append(0.0)
                    
                    if 'Es_Ciclico' in row:
                        frame_features.append(float(row['Es_Ciclico'] == 1))
                    else:
                        frame_features.append(0.0)
                    
                    sequence_data.append(frame_features)
            
            if len(sequence_data) == 0:
                return None, None, None
            
            # Convertir a numpy array
            sequence = np.array(sequence_data, dtype=np.float32)
            
            # Limpiar NaN e infinitos
            sequence = np.nan_to_num(sequence, nan=0.0, posinf=1e6, neginf=-1e6)
            
            return sequence, video_class, video_name
            
        except Exception as e:
            print(f"Error procesando {csv_path}: {e}")
            return None, None, None
    
    def load_videos_as_sequences(self, data_dir, max_sequence_length=100):
        """
        Carga todos los videos como secuencias temporales.
        """
        print("📁 Cargando videos como secuencias temporales...")
        
        data_dir = Path(data_dir)
        classes = ['normal', 'merodeo', 'forcejeo']
        
        all_sequences = []
        all_labels = []
        all_names = []
        
        for class_name in classes:
            class_dir = data_dir / class_name
            
            if not class_dir.exists():
                print(f"⚠️ Carpeta {class_name} no encontrada")
                continue
            
            csv_files = list(class_dir.glob("*.csv"))
            
            if not csv_files:
                print(f"⚠️ No hay archivos CSV en {class_dir}")
                continue
            
            print(f"📊 Clase '{class_name}': {len(csv_files)} videos")
            
            for csv_file in csv_files:
                sequence, label, name = self.extract_sequence_features(csv_file, class_name)
                
                if sequence is not None:
                    all_sequences.append(sequence)
                    all_labels.append(label)
                    all_names.append(name)
                    print(f"   ✅ {csv_file.name} ({len(sequence)} frames, {sequence.shape[1]} features)")
                else:
                    print(f"   ❌ {csv_file.name}")
        
        if not all_sequences:
            raise ValueError("No se procesaron secuencias")
        
        # Determinar longitud máxima de secuencia
        lengths = [len(seq) for seq in all_sequences]
        max_length = min(max_sequence_length, int(np.percentile(lengths, 90)))
        
        print(f"\n📊 Estadísticas de secuencias:")
        print(f"   Videos procesados: {len(all_sequences)}")
        print(f"   Longitud promedio: {np.mean(lengths):.1f} frames")
        print(f"   Longitud máxima usada: {max_length} frames")
        print(f"   Características por frame: {all_sequences[0].shape[1]}")
        
        # Padding/truncating de secuencias
        padded_sequences = []
        
        for seq in all_sequences:
            if len(seq) > max_length:
                # Truncar secuencia (tomar los primeros frames)
                padded_seq = seq[:max_length]
            elif len(seq) < max_length:
                # Hacer padding con ceros
                padding = np.zeros((max_length - len(seq), seq.shape[1]), dtype=np.float32)
                padded_seq = np.vstack([seq, padding])
            else:
                padded_seq = seq
            
            padded_sequences.append(padded_seq)
        
        # Convertir a arrays numpy
        self.sequences = np.array(padded_sequences, dtype=np.float32)
        self.labels = np.array(all_labels)
        self.video_names = np.array(all_names)
        
        print(f"   Forma final de datos: {self.sequences.shape}")
        print(f"   Distribución de clases: {dict(pd.Series(all_labels).value_counts())}")
        
        return self.sequences, self.labels, self.video_names
    

    def prepare_data_for_lstm(self, test_size=0.2):
        """
        Prepara datos para entrenamiento LSTM con separación por video completo.
        """
        print(f"\n🔄 Preparando datos para LSTM (con separación por video)...")

        y_encoded = self.label_encoder.fit_transform(self.labels)
        n_classes = len(self.label_encoder.classes_)
        print(f"   Clases: {dict(zip(self.label_encoder.classes_, range(n_classes)))}")

        # Separación por video completo usando GroupShuffleSplit
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=self.random_state)
        train_idx, test_idx = next(gss.split(self.sequences, y_encoded, groups=self.video_names))

        X_train, X_test = self.sequences[train_idx], self.sequences[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

        print(f"   ✅ División por video (sin fuga de datos)")
        print(f"   Train videos: {len(X_train)} | Test videos: {len(X_test)}")

        # Normalización
        n_samples_train, n_timesteps, n_features = X_train.shape
        X_train_reshaped = X_train.reshape(-1, n_features)
        self.scaler.fit(X_train_reshaped)
        X_train_scaled = self.scaler.transform(X_train_reshaped).reshape(n_samples_train, n_timesteps, n_features)

        n_samples_test = X_test.shape[0]
        X_test_reshaped = X_test.reshape(-1, n_features)
        X_test_scaled = self.scaler.transform(X_test_reshaped).reshape(n_samples_test, n_timesteps, n_features)

        y_train_cat = to_categorical(y_train, n_classes)
        y_test_cat = to_categorical(y_test, n_classes)

        self.X_train = X_train_scaled
        self.y_train = y_train
        self.y_train_cat = y_train_cat
        self.X_test = X_test_scaled
        self.y_test = y_test
        self.y_test_cat = y_test_cat
        self.n_classes = n_classes

        print(f"   Forma X_train: {X_train_scaled.shape}")
        print(f"   Forma y_train: {y_train_cat.shape}")

        return X_train_scaled, y_train_cat, X_test_scaled, y_test_cat


    def create_simple_lstm_model(self):
        """
        Modelo LSTM con configuración del segundo mejor resultado.
        """
        n_timesteps, n_features = self.X_train.shape[1], self.X_train.shape[2]
        n_classes = self.n_classes

        print(f"\n🏗️ Creando modelo LSTM (2 capas)...")
        print(f"   Timesteps: {n_timesteps}")
        print(f"   Features: {n_features}")
        print(f"   Clases: {n_classes}")

        model = Sequential()

        # Capa LSTM 1 (con return_sequences=True para permitir otra LSTM después)
        model.add(Bidirectional(LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
                                input_shape=(n_timesteps, n_features)))
        model.add(BatchNormalization())

        # Capa LSTM 2
        model.add(Bidirectional(LSTM(32, return_sequences=False, dropout=0.2, recurrent_dropout=0.2)))
        model.add(BatchNormalization())

        # Capas densas
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))

        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))

        # Capa de salida
        model.add(Dense(n_classes, activation='softmax'))

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        model.summary()
        return model


    
    def train_lstm(self, epochs=200, batch_size=8, validation_split=0.2):
        """
        Entrena el modelo LSTM con mejoras.
        """
        print(f"\n🏋️ Entrenando LSTM mejorado...")

        self.model = self.create_simple_lstm_model()

        if batch_size is None:
            batch_size = min(16, len(self.X_train) // 4)
            batch_size = max(2, batch_size)

        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Validation split: {validation_split}")

        # Calcular pesos de clase
        class_weights = compute_class_weight(class_weight='balanced',
                                            classes=np.unique(self.y_train),
                                            y=self.y_train)
        class_weight_dict = dict(enumerate(class_weights))

        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6,
                verbose=1
            )
        ]

        history = self.model.fit(
            self.X_train, self.y_train_cat,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=1
        )

        self.results['training_history'] = history.history
        print(f"   ✅ Entrenamiento completado")
        return history
    
    def train_lstm_from_config(self, cfg, epochs=150, validation_split=0.2):
        """
        Entrena el modelo LSTM usando un diccionario de hiperparámetros (cfg)
        tal como viene de grid_search_lstm.
        """
        print("\n🏗️  Re-entrenando con la MEJOR configuración de Grid Search…")
        print(f"   Config: {cfg}")

        n_timesteps, n_features = self.X_train.shape[1], self.X_train.shape[2]

        model = Sequential()
        model.add(Bidirectional(
            LSTM(cfg['lstm_units_1'],
                return_sequences=cfg['n_lstm_layers'] > 1,
                dropout=cfg['dropout'],
                recurrent_dropout=cfg['dropout']),
            input_shape=(n_timesteps, n_features)))
        model.add(BatchNormalization())

        if cfg['n_lstm_layers'] > 1:
            model.add(Bidirectional(
                LSTM(cfg.get('lstm_units_2', 32),
                    return_sequences=False,
                    dropout=cfg['dropout'],
                    recurrent_dropout=cfg['dropout'])))
            model.add(BatchNormalization())

        for units in cfg['dense_units']:
            model.add(Dense(units, activation='relu'))
            model.add(Dropout(cfg['dropout']))

        model.add(Dense(self.n_classes, activation='softmax'))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=cfg['learning_rate']),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        self.model = model
        print(model.summary())

        # Pesos por clase
        cw = compute_class_weight('balanced',
                                classes=np.unique(self.y_train),
                                y=self.y_train)
        cw_dict = dict(enumerate(cw))

        cb = [
            EarlyStopping(monitor='val_loss', patience=20,
                        restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                            patience=10, min_lr=1e-6, verbose=1)
        ]

        history = self.model.fit(
            self.X_train, self.y_train_cat,
            epochs=epochs,
            batch_size=cfg['batch_size'],
            validation_split=validation_split,
            callbacks=cb,
            class_weight=cw_dict,
            verbose=1
        )

        self.results['training_history'] = history.history
        print("✅ Re-entrenamiento completado")
        return history

    
    def evaluate_final(self):
        """
        Evaluación final en datos de test.
        """
        print(f"\n🎯 EVALUACIÓN FINAL...")
        
        # Predicciones
        y_pred_proba = self.model.predict(self.X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Métricas
        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        
        print(f"   Videos test: {len(self.y_test)}")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   F1-Score: {f1:.4f}")
        
        # Reporte detallado
        class_names = self.label_encoder.classes_
        report = classification_report(self.y_test, y_pred, target_names=class_names)
        print(f"\n📋 Reporte por clase:")
        print(report)
        
        # Matriz de confusión
        cm = confusion_matrix(self.y_test, y_pred)
        
        # Guardar resultados
        self.results.update({
            'final_accuracy': accuracy,
            'final_f1': f1,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'prediction_probabilities': y_pred_proba
        })
        
        return accuracy, f1
    
    def plot_results(self):
        """
        Visualiza resultados del entrenamiento.
        """
        if 'training_history' not in self.results:
            print("⚠️ No hay historia de entrenamiento")
            return
        
        history = self.results['training_history']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Loss
        axes[0, 0].plot(history['loss'], label='Train Loss')
        if 'val_loss' in history:
            axes[0, 0].plot(history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Pérdida durante entrenamiento')
        axes[0, 0].set_xlabel('Época')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 2. Accuracy
        axes[0, 1].plot(history['accuracy'], label='Train Accuracy')
        if 'val_accuracy' in history:
            axes[0, 1].plot(history['val_accuracy'], label='Val Accuracy')
        axes[0, 1].set_title('Precisión durante entrenamiento')
        axes[0, 1].set_xlabel('Época')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 3. Matriz de confusión
        if 'confusion_matrix' in self.results:
            cm = self.results['confusion_matrix']
            class_names = self.label_encoder.classes_
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names,
                       ax=axes[1, 0])
            axes[1, 0].set_title('Matriz de Confusión')
            axes[1, 0].set_ylabel('Real')
            axes[1, 0].set_xlabel('Predicho')
        
        # 4. Distribución de predicciones
        if 'prediction_probabilities' in self.results:
            probs = self.results['prediction_probabilities']
            max_probs = np.max(probs, axis=1)
            
            axes[1, 1].hist(max_probs, bins=20, alpha=0.7, edgecolor='black')
            axes[1, 1].set_title('Distribución de Confianza en Predicciones')
            axes[1, 1].set_xlabel('Probabilidad Máxima')
            axes[1, 1].set_ylabel('Frecuencia')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath='simple_lstm_model.h5'):
        """
        Guarda el modelo entrenado.
        """
        if self.model is not None:
            self.model.save(filepath)
            
            # Guardar también scaler y label encoder
            import pickle
            metadata = {
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'sequence_shape': self.X_train.shape,
                'classes': self.label_encoder.classes_.tolist()
            }
            
            metadata_path = filepath.replace('.h5', '_metadata.pkl')
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            print(f"💾 Modelo guardado: {filepath}")
            print(f"💾 Metadatos guardados: {metadata_path}")

def plot_test_results(y_test, y_pred, y_prob, label_encoder):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.metrics import confusion_matrix

    class_names = label_encoder.classes_
    df = pd.DataFrame({
        'Real': [class_names[i] for i in y_test],
        'Predicho': [class_names[i] for i in y_pred],
        'Confianza': [round(probs[pred], 2) for probs, pred in zip(y_prob, y_pred)]
    })

    # Gráfica 1: Matriz de Confusión
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Matriz de Confusión (Test)")
    plt.ylabel("Clase Real")
    plt.xlabel("Clase Predicha")
    plt.tight_layout()
    plt.show()

    # Gráfica 2: Barras Real vs Predicho
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='Real', hue='Predicho', palette='Set2')
    plt.title("Distribución de Predicciones en el Test")
    plt.ylabel("Cantidad")
    plt.xlabel("Clase Real")
    plt.legend(title="Predicho")
    plt.tight_layout()
    plt.show()

    # Gráfica 3: Confianza del modelo
    plt.figure(figsize=(6, 5))
    sns.histplot(df['Confianza'], bins=10, kde=True, color='purple')
    plt.title("Confianza en las Predicciones")
    plt.xlabel("Confianza")
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.show()


def main():
    print("🔄 ENTRENADOR LSTM SIMPLE PARA VIDEOS")
    print("=" * 50)

    # data_dir = input("📁 Directorio con carpetas de clases: ").strip()
    data_dir = r'D:\Documentos\Monitoreo-Comportamientos-Sospechosos\csv'
    if not data_dir or not Path(data_dir).exists():
        print("❌ Directorio no válido"); return

    trainer = SimpleVideoLSTMTrainer(random_state=42)

    # 1. Cargar y preparar
    trainer.load_videos_as_sequences(data_dir, max_sequence_length=50)
    trainer.prepare_data_for_lstm(test_size=0.2)

    # 2. ¿Grid Search?
    ejecutar_grid = input("⚙️ ¿Ejecutar Grid Search de hiperparámetros? [s/N]: ").strip().lower()

    if ejecutar_grid == 's':
        print("\n🚀 Ejecutando SOLO Grid Search (no habrá entrenamiento por defecto)…")

        param_grid = {
            'lstm_units_1': [64],
            'lstm_units_2': [32],
            'n_lstm_layers': [1,2],
            'dense_units': [(64, 32)],
            'dropout': [0.2],
            'learning_rate': [ 0.0005],
            'batch_size': [8]
        }

        top_results = grid_search_lstm(
            trainer,
            param_grid,
            epochs=150,
            validation_split=0.2
        )

        # 2.1 Elegir la mejor config (top_results[0])
        best_cfg = top_results[0]['config']
        print(f"\n✨ Mejor configuración encontrada: {best_cfg}")

        # 2.2 Re-entrenar con la mejor config y guardar historia para gráficas
        history = trainer.train_lstm_from_config(best_cfg, epochs=150, validation_split=0.2)

    else:
        print("\n🚀 Entrenamiento con hiperparámetros por defecto…")
        history = trainer.train_lstm(epochs=200, validation_split=0.2)

    # 3. Evaluar y graficar
    trainer.evaluate_final()
    trainer.plot_results()
    plot_test_results(trainer.y_test,
                      trainer.results['predictions'],
                      trainer.results['prediction_probabilities'],
                      trainer.label_encoder)


class SimpleLSTMPredictor:
    """
    Predictor para usar modelo LSTM entrenado.
    """
    
    def __init__(self, model_path, metadata_path):
        """
        Carga modelo y metadatos.
        """
        import pickle
        
        self.model = tf.keras.models.load_model(model_path)
        
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        self.scaler = metadata['scaler']
        self.label_encoder = metadata['label_encoder']
        self.sequence_shape = metadata['sequence_shape']
        
        print(f"✅ Modelo LSTM cargado")
        print(f"   Clases: {metadata['classes']}")
    
    def predict_video(self, csv_path):
        """
        Predice clase de un nuevo video.
        """
        # Crear trainer temporal para extraer secuencia
        temp_trainer = SimpleVideoLSTMTrainer()
        sequence, _, _ = temp_trainer.extract_sequence_features(csv_path, "dummy")
        
        if sequence is None:
            raise ValueError(f"No se pudo procesar {csv_path}")
        
        # Ajustar longitud de secuencia
        target_length = self.sequence_shape[1]
        
        if len(sequence) > target_length:
            sequence = sequence[:target_length]
        elif len(sequence) < target_length:
            padding = np.zeros((target_length - len(sequence), sequence.shape[1]))
            sequence = np.vstack([sequence, padding])
        
        # Normalizar
        sequence_reshaped = sequence.reshape(-1, sequence.shape[1])
        sequence_scaled = self.scaler.transform(sequence_reshaped)
        sequence_final = sequence_scaled.reshape(1, target_length, sequence.shape[1])
        
        # Predecir
        prediction_proba = self.model.predict(sequence_final, verbose=0)[0]
        prediction_class_idx = np.argmax(prediction_proba)
        prediction_class = self.label_encoder.inverse_transform([prediction_class_idx])[0]
        
        # Crear diccionario de probabilidades
        class_probabilities = {}
        for i, class_name in enumerate(self.label_encoder.classes_):
            class_probabilities[class_name] = float(prediction_proba[i])
        
        return prediction_class, class_probabilities

def grid_search_lstm(trainer, param_grid, epochs=150, validation_split=0.2):
    import itertools

    print("\n🔍 Iniciando Grid Search para LSTM...")
    results = []
    combinations = list(itertools.product(*param_grid.values()))

    for i, combo in enumerate(combinations):
        config = dict(zip(param_grid.keys(), combo))
        print(f"\n🔧 Configuración {i+1}/{len(combinations)}: {config}")

        # Crear modelo con la configuración
        model = Sequential()
        input_shape = (trainer.X_train.shape[1], trainer.X_train.shape[2])

        # Capa LSTM 1
        model.add(Bidirectional(LSTM(config['lstm_units_1'], return_sequences=config['n_lstm_layers'] > 1, 
                                     dropout=config['dropout'], recurrent_dropout=config['dropout']), 
                                input_shape=input_shape))
        model.add(BatchNormalization())

        # Capa LSTM 2 opcional
        if config['n_lstm_layers'] > 1:
            model.add(Bidirectional(LSTM(config['lstm_units_2'], return_sequences=False, 
                                         dropout=config['dropout'], recurrent_dropout=config['dropout'])))
            model.add(BatchNormalization())

        # Capas densas
        for units in config['dense_units']:
            model.add(Dense(units, activation='relu'))
            model.add(Dropout(config['dropout']))

        model.add(Dense(trainer.n_classes, activation='softmax'))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config['learning_rate']),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=0)
        ]

        # Calcular class weights
        class_weights = compute_class_weight(class_weight='balanced',
                                             classes=np.unique(trainer.y_train),
                                             y=trainer.y_train)
        class_weight_dict = dict(enumerate(class_weights))

        # Entrenar
        history = model.fit(
            trainer.X_train, trainer.y_train_cat,
            epochs=epochs,
            batch_size=config['batch_size'],
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=0,
            class_weight=class_weight_dict
        )

        # Evaluar
        y_pred = np.argmax(model.predict(trainer.X_test, verbose=0), axis=1)
        acc = accuracy_score(trainer.y_test, y_pred)
        f1 = f1_score(trainer.y_test, y_pred, average='weighted')

        results.append({
            'config': config,
            'accuracy': acc,
            'f1_score': f1
        })
        print(f"✔️  Finalizado modelo {i+1}/{len(combinations)} "
        f"→ Acc={acc:.3f} | F1={f1:.3f}")


        print(f"✅ Accuracy: {acc:.4f}, F1: {f1:.4f}")
    
    print("\n🔚 Grid Search terminado.")

    # Mostrar top resultados
    top_results = sorted(results, key=lambda x: x['f1_score'], reverse=True)[:3]
    print("\n🏆 Mejores configuraciones:")
    for r in top_results:
        print(f"F1={r['f1_score']:.4f} | Acc={r['accuracy']:.4f} | Config: {r['config']}")

    return top_results

if __name__ == "__main__":
    main()




