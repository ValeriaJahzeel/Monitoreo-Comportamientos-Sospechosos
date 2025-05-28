"""
Entrenador MLP para clasificación de videos basado en características agregadas.
Diseñado para 49 videos (archivos CSV) distribuidos en carpetas por clase.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

class VideoLevelMLPTrainer:
    """
    Entrenador MLP para clasificación de videos usando características agregadas.
    Cada video (archivo CSV) es una muestra.
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model = None
        self.video_features = None
        self.results = {}
        
    def extract_video_features(self, csv_path, video_class):
        """
        Extrae características agregadas de un video (archivo CSV).
        
        Parámetros:
        -----------
        csv_path : str
            Ruta al archivo CSV del video
        video_class : str
            Clase del video (normal, merodeo, forcejeo)
        """
        try:
            df = pd.read_csv(csv_path)
            
            if df.empty:
                return None
                
            video_name = Path(csv_path).stem
            
            # Características numéricas para agregar
            numeric_cols = ['Desplazamiento', 'Velocidad', 'Aceleracion', 'Direccion', 
                          'Densidad', 'Linealidad', 'Circularidad', 'Zigzag', 
                          'Frecuencia_Ciclo', 'Amplitud_Ciclo', 'Area_Trayectoria']
            
            # Verificar qué columnas existen
            available_numeric = [col for col in numeric_cols if col in df.columns]
            
            features = {'video_id': video_name, 'clase': video_class}
            
            # 1. CARACTERÍSTICAS ESTADÍSTICAS BÁSICAS
            for col in available_numeric:
                if not df[col].isna().all():
                    data = df[col].dropna()
                    if len(data) > 0:
                        features[f'{col}_mean'] = data.mean()
                        features[f'{col}_max'] = data.max()
                        features[f'{col}_min'] = data.min()
                        features[f'{col}_std'] = data.std() if len(data) > 1 else 0
                        features[f'{col}_median'] = data.median()
            
            # 2. CARACTERÍSTICAS TEMPORALES
            features['duracion_video'] = len(df)
            features['frames_unicos'] = df['Frame'].nunique() if 'Frame' in df.columns else len(df)
            features['objetos_unicos'] = df['Objeto'].nunique() if 'Objeto' in df.columns else 1
            
            # 3. CARACTERÍSTICAS DE COMPORTAMIENTO
            # Interacciones
            if 'En_Interaccion' in df.columns:
                features['porcentaje_interaccion'] = (df['En_Interaccion'] == 1).mean()
                features['total_interacciones'] = (df['En_Interaccion'] == 1).sum()
            
            # Ciclos
            if 'Es_Ciclico' in df.columns:
                features['porcentaje_ciclico'] = (df['Es_Ciclico'] == 1).mean()
                features['total_ciclico'] = (df['Es_Ciclico'] == 1).sum()
            
            # 4. CARACTERÍSTICAS DE POSTURA Y MOVIMIENTO
            if 'Postura' in df.columns:
                postura_counts = df['Postura'].value_counts(normalize=True)
                for postura, freq in postura_counts.items():
                    features[f'postura_{postura}_freq'] = freq
            
            if 'Patron_Movimiento' in df.columns:
                patron_counts = df['Patron_Movimiento'].value_counts(normalize=True)
                for patron, freq in patron_counts.items():
                    features[f'patron_{patron}_freq'] = freq
            
            # 5. CARACTERÍSTICAS ESPACIALES
            if 'Centroide_X' in df.columns and 'Centroide_Y' in df.columns:
                features['rango_x'] = df['Centroide_X'].max() - df['Centroide_X'].min()
                features['rango_y'] = df['Centroide_Y'].max() - df['Centroide_Y'].min()
                features['centro_x'] = df['Centroide_X'].mean()
                features['centro_y'] = df['Centroide_Y'].mean()
                
                # Variabilidad espacial
                features['variabilidad_x'] = df['Centroide_X'].std()
                features['variabilidad_y'] = df['Centroide_Y'].std()
            
            # 6. CARACTERÍSTICAS AVANZADAS
            # Cambios bruscos en velocidad (posible indicador de comportamiento anómalo)
            if 'Velocidad' in df.columns:
                velocidad = df['Velocidad'].dropna()
                if len(velocidad) > 1:
                    vel_diff = np.abs(np.diff(velocidad))
                    features['cambios_velocidad_mean'] = vel_diff.mean()
                    features['cambios_velocidad_max'] = vel_diff.max()
                    features['velocidad_cero_freq'] = (velocidad == 0).mean()
            
            # Variabilidad en dirección
            if 'Direccion' in df.columns:
                direccion = df['Direccion'].dropna()
                if len(direccion) > 1:
                    dir_diff = np.abs(np.diff(direccion))
                    features['cambios_direccion_mean'] = dir_diff.mean()
                    features['cambios_direccion_max'] = dir_diff.max()
            
            return features
            
        except Exception as e:
            print(f"Error procesando {csv_path}: {e}")
            return None
    
    def load_videos_from_folders(self, data_dir):
        """
        Carga todos los videos desde carpetas organizadas por clase.
        
        Parámetros:
        -----------
        data_dir : str
            Directorio base que contiene carpetas: normal/, merodeo/, forcejeo/
        """
        print("📁 Cargando videos desde carpetas...")
        
        data_dir = Path(data_dir)
        classes = ['normal', 'merodeo', 'forcejeo']
        
        all_features = []
        
        for class_name in classes:
            class_dir = data_dir / class_name
            
            if not class_dir.exists():
                print(f"⚠️ Carpeta {class_name} no encontrada en {data_dir}")
                continue
            
            # Buscar archivos CSV
            csv_files = list(class_dir.glob("*.csv"))
            
            if not csv_files:
                print(f"⚠️ No se encontraron archivos CSV en {class_dir}")
                continue
            
            print(f"📊 Clase '{class_name}': {len(csv_files)} videos")
            
            for csv_file in csv_files:
                features = self.extract_video_features(csv_file, class_name)
                if features is not None:
                    all_features.append(features)
                    print(f"   ✅ {csv_file.name}")
                else:
                    print(f"   ❌ {csv_file.name}")
        
        if not all_features:
            raise ValueError("No se pudieron extraer características de ningún video")
        
        # Convertir a DataFrame
        self.video_features = pd.DataFrame(all_features)
        
        # Rellenar NaN con 0 (características que no todos los videos tienen)
        numeric_columns = self.video_features.select_dtypes(include=[np.number]).columns
        self.video_features[numeric_columns] = self.video_features[numeric_columns].fillna(0)
        
        print(f"\n📊 Resumen de carga:")
        print(f"   Total videos procesados: {len(self.video_features)}")
        print(f"   Características extraídas: {len(numeric_columns)}")
        print(f"   Distribución por clase:")
        print(self.video_features['clase'].value_counts())
        
        return self.video_features
    
    def prepare_data_for_training(self, test_size=0.2):
        """
        Prepara datos para entrenamiento, separando videos de test.
        """
        if self.video_features is None:
            raise ValueError("Primero debe cargar los videos con load_videos_from_folders()")
        
        print(f"\n🔄 Preparando datos para entrenamiento...")
        
        # Separar características y objetivo
        exclude_cols = ['video_id', 'clase']
        feature_cols = [col for col in self.video_features.columns if col not in exclude_cols]
        
        X = self.video_features[feature_cols]
        y = self.video_features['clase']
        
        print(f"   Videos totales: {len(X)}")
        print(f"   Características: {len(feature_cols)}")
        
        # Codificar clases
        y_encoded = self.label_encoder.fit_transform(y)
        class_mapping = dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))
        print(f"   Clases codificadas: {class_mapping}")
        
        # División estratificada
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=test_size, 
                stratify=y_encoded, random_state=self.random_state
            )
            print(f"   ✅ División estratificada exitosa")
        except ValueError as e:
            print(f"   ⚠️ División estratificada falló: {e}")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=test_size, random_state=self.random_state
            )
            print(f"   ✅ División simple aplicada")
        
        # Mostrar distribución
        print(f"\n   Distribución final:")
        print(f"   Train: {len(X_train)} videos")
        for class_idx, class_name in enumerate(self.label_encoder.classes_):
            count = (y_train == class_idx).sum()
            print(f"     {class_name}: {count}")
        
        print(f"   Test: {len(X_test)} videos")
        for class_idx, class_name in enumerate(self.label_encoder.classes_):
            count = (y_test == class_idx).sum()
            print(f"     {class_name}: {count}")
        
        # Escalar características
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Guardar datos
        self.X_train = X_train_scaled
        self.y_train = y_train
        self.X_test = X_test_scaled
        self.y_test = y_test
        self.feature_names = feature_cols
        
        return X_train_scaled, y_train, X_test_scaled, y_test
    
    def get_mlp_configurations(self, n_samples):
        """
        Define configuraciones MLP optimizadas para el número de videos disponible.
        """
        if n_samples < 30:
            # Muy pocos videos - arquitecturas muy simples
            configs = [
                {
                    'hidden_layer_sizes': (4,),
                    'activation': 'relu',
                    'alpha': 1.0,
                    'learning_rate': 'adaptive',
                    'learning_rate_init': 0.01,
                    'max_iter': 3000,
                    'early_stopping': True,
                    'validation_fraction': 0.3,
                    'n_iter_no_change': 30,
                    'random_state': self.random_state
                },
                {
                    'hidden_layer_sizes': (8,),
                    'activation': 'tanh',
                    'alpha': 0.5,
                    'learning_rate': 'adaptive',
                    'learning_rate_init': 0.01,
                    'max_iter': 3000,
                    'early_stopping': True,
                    'validation_fraction': 0.3,
                    'n_iter_no_change': 30,
                    'random_state': self.random_state
                },
                {
                    'hidden_layer_sizes': (6, 3),
                    'activation': 'relu',
                    'alpha': 0.5,
                    'learning_rate': 'adaptive',
                    'learning_rate_init': 0.005,
                    'max_iter': 3000,
                    'early_stopping': True,
                    'validation_fraction': 0.3,
                    'n_iter_no_change': 30,
                    'random_state': self.random_state
                }
            ]
        elif n_samples < 50:
            # Pocos videos - arquitecturas conservadoras
            configs = [
                {
                    'hidden_layer_sizes': (8,),
                    'activation': 'relu',
                    'alpha': 0.1,
                    'learning_rate': 'adaptive',
                    'learning_rate_init': 0.01,
                    'max_iter': 2000,
                    'early_stopping': True,
                    'validation_fraction': 0.25,
                    'n_iter_no_change': 25,
                    'random_state': self.random_state
                },
                {
                    'hidden_layer_sizes': (16,),
                    'activation': 'tanh',
                    'alpha': 0.1,
                    'learning_rate': 'adaptive',
                    'learning_rate_init': 0.01,
                    'max_iter': 2000,
                    'early_stopping': True,
                    'validation_fraction': 0.25,
                    'n_iter_no_change': 25,
                    'random_state': self.random_state
                },
                {
                    'hidden_layer_sizes': (12, 6),
                    'activation': 'relu',
                    'alpha': 0.05,
                    'learning_rate': 'adaptive',
                    'learning_rate_init': 0.005,
                    'max_iter': 2000,
                    'early_stopping': True,
                    'validation_fraction': 0.25,
                    'n_iter_no_change': 25,
                    'random_state': self.random_state
                },
                {
                    'hidden_layer_sizes': (16, 8),
                    'activation': 'tanh',
                    'alpha': 0.01,
                    'learning_rate': 'adaptive',
                    'learning_rate_init': 0.001,
                    'max_iter': 2000,
                    'early_stopping': True,
                    'validation_fraction': 0.25,
                    'n_iter_no_change': 25,
                    'random_state': self.random_state
                }
            ]
        else:
            # Suficientes videos - arquitecturas más complejas
            configs = [
                {
                    'hidden_layer_sizes': (16,),
                    'activation': 'relu',
                    'alpha': 0.01,
                    'learning_rate': 'adaptive',
                    'learning_rate_init': 0.001,
                    'max_iter': 1500,
                    'early_stopping': True,
                    'validation_fraction': 0.2,
                    'n_iter_no_change': 20,
                    'random_state': self.random_state
                },
                {
                    'hidden_layer_sizes': (32,),
                    'activation': 'relu',
                    'alpha': 0.01,
                    'learning_rate': 'adaptive',
                    'learning_rate_init': 0.001,
                    'max_iter': 1500,
                    'early_stopping': True,
                    'validation_fraction': 0.2,
                    'n_iter_no_change': 20,
                    'random_state': self.random_state
                },
                {
                    'hidden_layer_sizes': (24, 12),
                    'activation': 'tanh',
                    'alpha': 0.005,
                    'learning_rate': 'adaptive',
                    'learning_rate_init': 0.001,
                    'max_iter': 1500,
                    'early_stopping': True,
                    'validation_fraction': 0.2,
                    'n_iter_no_change': 20,
                    'random_state': self.random_state
                }
            ]
        
        return configs
    
    def balance_classes_manually(self, X, y):
        """
        Balancea clases manualmente usando sobremuestreo para la clase minoritaria.
        """
        from collections import Counter
        
        class_counts = Counter(y)
        max_count = max(class_counts.values())
        
        # Si las clases ya están balanceadas, no hacer nada
        if min(class_counts.values()) / max_count > 0.7:  # Diferencia menor al 30%
            return X, y
        
        print(f"   🔄 Balanceando clases manualmente:")
        for class_label, count in class_counts.items():
            print(f"      Clase {class_label}: {count} muestras")
        
        X_balanced = []
        y_balanced = []
        
        # Para cada clase, duplicar muestras hasta alcanzar el máximo
        for class_label in class_counts.keys():
            class_mask = (y == class_label)
            class_X = X[class_mask]
            class_y = y[class_mask]
            
            current_count = len(class_X)
            target_count = max_count
            
            # Agregar muestras originales
            X_balanced.extend(class_X)
            y_balanced.extend(class_y)
            
            # Sobremuestrear si es necesario
            if current_count < target_count:
                samples_needed = target_count - current_count
                
                # Seleccionar muestras aleatorias para duplicar
                np.random.seed(self.random_state)
                indices_to_duplicate = np.random.choice(
                    len(class_X), size=samples_needed, replace=True
                )
                
                for idx in indices_to_duplicate:
                    X_balanced.append(class_X[idx])
                    y_balanced.append(class_y[idx])
        
        X_balanced = np.array(X_balanced)
        y_balanced = np.array(y_balanced)
        
        print(f"   ✅ Datos balanceados: {len(X_balanced)} muestras totales")
        
        return X_balanced, y_balanced
    
    def train_with_cross_validation(self, cv_folds=3, balance_classes=True):
        """
        Entrena MLP probando múltiples configuraciones con validación cruzada.
        """
        if self.X_train is None:
            raise ValueError("Primero debe preparar los datos con prepare_data_for_training()")
        
        print(f"\n🧠 Entrenando MLP con validación cruzada...")
        
        n_samples = len(self.X_train)
        n_features = self.X_train.shape[1]
        n_classes = len(np.unique(self.y_train))
        
        print(f"   Videos entrenamiento: {n_samples}")
        print(f"   Características: {n_features}")
        print(f"   Clases: {n_classes}")
        
        # Obtener configuraciones
        configs = self.get_mlp_configurations(n_samples)
        
        # Ajustar CV según datos disponibles
        max_cv_folds = n_samples // 5  # Mínimo 5 videos por fold
        cv_folds = min(cv_folds, max_cv_folds, 5)  # Máximo 5 folds
        cv_folds = max(cv_folds, 2)  # Mínimo 2 folds
        
        print(f"   Configuraciones a probar: {len(configs)}")
        print(f"   Folds de CV: {cv_folds}")
        
        best_score = -1
        best_config = None
        best_model = None
        cv_results = []
        
        for i, config in enumerate(configs):
            try:
                print(f"\n   🔄 Probando configuración {i+1}/{len(configs)}")
                print(f"      Arquitectura: {config['hidden_layer_sizes']}")
                print(f"      Activación: {config['activation']}")
                print(f"      Alpha: {config['alpha']}")
                
                mlp = MLPClassifier(**config)
                
                # Validación cruzada estratificada
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
                scores = cross_val_score(mlp, self.X_train, self.y_train, 
                                       cv=cv, scoring='f1_weighted')
                
                mean_score = scores.mean()
                std_score = scores.std()
                
                cv_results.append({
                    'config': i + 1,
                    'architecture': str(config['hidden_layer_sizes']),
                    'activation': config['activation'],
                    'alpha': config['alpha'],
                    'mean_f1': mean_score,
                    'std_f1': std_score,
                    'scores': scores
                })
                
                print(f"      F1-Score: {mean_score:.4f} (±{std_score:.4f})")
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_config = config
                    best_model = mlp
                    print(f"      🏆 Nueva mejor configuración!")
                
            except Exception as e:
                print(f"      ❌ Error: {str(e)}")
        
        if best_model is None:
            raise ValueError("No se pudo entrenar ningún modelo")
        
        print(f"\n🏆 Mejor configuración encontrada:")
        print(f"   F1-Score CV: {best_score:.4f}")
        print(f"   Arquitectura: {best_config['hidden_layer_sizes']}")
        print(f"   Activación: {best_config['activation']}")
        print(f"   Alpha: {best_config['alpha']}")
        
        # Entrenar modelo final con todos los datos de entrenamiento
        print(f"\n🔄 Entrenando modelo final...")
        
        # Balancear clases manualmente si se solicita
        X_final = self.X_train
        y_final = self.y_train
        
        if balance_classes:
            X_final, y_final = self.balance_classes_manually(self.X_train, self.y_train)
        
        # Crear y entrenar modelo final
        self.model = MLPClassifier(**best_config)
        self.model.fit(X_final, y_final)
        
        # Guardar resultados
        self.results = {
            'best_config': best_config,
            'best_cv_score': best_score,
            'cv_results': cv_results,
            'n_training_videos': n_samples,
            'cv_folds': cv_folds
        }
        
        print(f"   ✅ Modelo entrenado exitosamente")
        
        return self.model
    
    def evaluate_final_performance(self):
        """
        Evalúa el modelo en videos de test completamente no vistos.
        """
        if self.model is None:
            raise ValueError("Modelo no entrenado")
        
        print(f"\n🎯 EVALUACIÓN FINAL EN VIDEOS NO VISTOS...")
        
        # Predicciones
        y_pred = self.model.predict(self.X_test)
        y_pred_proba = self.model.predict_proba(self.X_test)
        
        # Métricas
        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        
        print(f"   📊 Resultados finales:")
        print(f"      Videos de test: {len(self.y_test)}")
        print(f"      Accuracy: {accuracy:.4f}")
        print(f"      F1-Score: {f1:.4f}")
        
        # Reporte detallado
        print(f"\n   📋 Reporte por clase:")
        class_names = self.label_encoder.classes_
        report = classification_report(self.y_test, y_pred, 
                                     target_names=class_names, 
                                     zero_division=0)
        print(report)
        
        # Matriz de confusión
        cm = confusion_matrix(self.y_test, y_pred)
        
        # Videos mal clasificados
        print(f"\n   🔍 Análisis de errores:")
        if len(self.y_test) == len(y_pred):
            # Obtener índices de videos de test
            test_videos = self.video_features.iloc[-len(self.y_test):]['video_id'].values
            
            for i, (true_label, pred_label) in enumerate(zip(self.y_test, y_pred)):
                if true_label != pred_label:
                    true_class = class_names[true_label]
                    pred_class = class_names[pred_label]
                    confidence = y_pred_proba[i][pred_label]
                    print(f"      {test_videos[i]}: {true_class} → {pred_class} (conf: {confidence:.3f})")
        
        # Guardar resultados
        self.results.update({
            'final_accuracy': accuracy,
            'final_f1': f1,
            'confusion_matrix': cm,
            'classification_report': report,
            'test_predictions': y_pred,
            'test_probabilities': y_pred_proba,
            'test_size': len(self.y_test)
        })
        
        return accuracy, f1
    
    def plot_results(self, save_path=None):
        """
        Genera visualizaciones de los resultados.
        """
        if not self.results:
            print("⚠️ No hay resultados para visualizar")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Resultados de validación cruzada
        if 'cv_results' in self.results:
            cv_df = pd.DataFrame(self.results['cv_results'])
            
            x_pos = range(len(cv_df))
            axes[0, 0].bar(x_pos, cv_df['mean_f1'], yerr=cv_df['std_f1'], 
                          capsize=5, alpha=0.7, color='skyblue')
            axes[0, 0].set_xlabel('Configuración')
            axes[0, 0].set_ylabel('F1-Score')
            axes[0, 0].set_title('Validación Cruzada por Configuración')
            axes[0, 0].set_xticks(x_pos)
            axes[0, 0].set_xticklabels([f"C{i}" for i in cv_df['config']])
            axes[0, 0].grid(True, alpha=0.3)
            
            # Resaltar mejor configuración
            best_idx = cv_df['mean_f1'].idxmax()
            axes[0, 0].bar(best_idx, cv_df.loc[best_idx, 'mean_f1'], 
                          color='red', alpha=0.8)
        
        # 2. Matriz de confusión
        if 'confusion_matrix' in self.results:
            cm = self.results['confusion_matrix']
            class_names = self.label_encoder.classes_
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names,
                       ax=axes[0, 1])
            axes[0, 1].set_title('Matriz de Confusión (Videos Test)')
            axes[0, 1].set_ylabel('Clase Real')
            axes[0, 1].set_xlabel('Clase Predicha')
        
        # 3. Comparación CV vs Test
        if 'best_cv_score' in self.results and 'final_f1' in self.results:
            metrics = ['F1 CV', 'F1 Test']
            values = [self.results['best_cv_score'], self.results['final_f1']]
            colors = ['blue', 'red']
            
            bars = axes[1, 0].bar(metrics, values, color=colors, alpha=0.7)
            axes[1, 0].set_ylabel('F1-Score')
            axes[1, 0].set_title('Comparación: Validación vs Test')
            axes[1, 0].set_ylim(0, 1)
            
            # Agregar valores
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                               f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Distribución de datos
        if hasattr(self, 'video_features'):
            class_counts = self.video_features['clase'].value_counts()
            
            wedges, texts, autotexts = axes[1, 1].pie(class_counts.values, 
                                                     labels=class_counts.index,
                                                     autopct='%1.1f%%',
                                                     startangle=90)
            axes[1, 1].set_title(f'Distribución de Videos por Clase\n(Total: {len(self.video_features)})')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Gráficos guardados en: {save_path}")
        
        plt.show()
    
    def get_feature_importance(self, top_n=15):
        """
        Calcula y muestra las características más importantes.
        """
        if self.model is None or not hasattr(self.model, 'coefs_'):
            print("⚠️ Modelo no disponible o sin coeficientes")
            return None
        
        # Calcular importancia aproximada
        first_layer_weights = np.abs(self.model.coefs_[0])
        importance_scores = np.mean(first_layer_weights, axis=1)
        
        # Crear DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names[:len(importance_scores)],
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        print(f"\n🔍 Top {top_n} características más importantes:")
        for i, (_, row) in enumerate(importance_df.head(top_n).iterrows()):
            print(f"   {i+1:2d}. {row['feature']:<25}: {row['importance']:.4f}")
        
        # Visualizar importancia
        plt.figure(figsize=(10, 8))
        top_features = importance_df.head(top_n)
        
        plt.barh(range(len(top_features)), top_features['importance'], color='skyblue')
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importancia')
        plt.title(f'Top {top_n} Características Más Importantes')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return importance_df
    
    def save_results(self, output_dir='./resultados_mlp'):
        """
        Guarda todos los resultados del entrenamiento.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Guardar características de videos
        if self.video_features is not None:
            self.video_features.to_csv(output_dir / 'video_features.csv', index=False)
        
        # Guardar configuración del modelo
        if self.results:
            import json
            
            # Preparar datos para JSON (sin objetos no serializables)
            results_to_save = self.results.copy()
            if 'confusion_matrix' in results_to_save:
                results_to_save['confusion_matrix'] = results_to_save['confusion_matrix'].tolist()
            if 'test_predictions' in results_to_save:
                results_to_save['test_predictions'] = results_to_save['test_predictions'].tolist()
            if 'test_probabilities' in results_to_save:
                results_to_save['test_probabilities'] = results_to_save['test_probabilities'].tolist()
            
            # Limpiar cv_results
            if 'cv_results' in results_to_save:
                for cv_result in results_to_save['cv_results']:
                    if 'scores' in cv_result:
                        cv_result['scores'] = cv_result['scores'].tolist()
            
            with open(output_dir / 'training_results.json', 'w') as f:
                json.dump(results_to_save, f, indent=2)
        
        # Guardar modelo
        if self.model is not None:
            import pickle
            
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'feature_names': self.feature_names
            }
            
            with open(output_dir / 'trained_model.pkl', 'wb') as f:
                pickle.dump(model_data, f)
        
        print(f"💾 Resultados guardados en: {output_dir}")

def main():
    """
    Función principal para ejecutar el entrenamiento.
    """
    print("🎬 ENTRENADOR MLP PARA CLASIFICACIÓN DE VIDEOS")
    print("=" * 60)
    print("📁 Estructura esperada:")
    print("   data_dir/")
    print("   ├── normal/")
    print("   │   ├── video1.csv")
    print("   │   └── video2.csv")
    print("   ├── merodeo/")
    print("   │   ├── video3.csv")
    print("   │   └── video4.csv")
    print("   └── forcejeo/")
    print("       ├── video5.csv")
    print("       └── video6.csv")
    print()
    
    # Solicitar directorio de datos
    data_dir = input("📁 Directorio con carpetas de clases: ").strip()
    if not data_dir:
        print("❌ Debe proporcionar el directorio de datos")
        return
    
    if not Path(data_dir).exists():
        print(f"❌ Directorio no encontrado: {data_dir}")
        return
    
    try:
        # Crear entrenador
        trainer = VideoLevelMLPTrainer(random_state=42)
        
        # Cargar videos
        video_features = trainer.load_videos_from_folders(data_dir)
        
        if len(video_features) < 10:
            print("⚠️ Muy pocos videos para entrenamiento confiable")
            continue_anyway = input("¿Continuar de todos modos? [s/N]: ").strip().lower()
            if continue_anyway != 's':
                return
        
        # Preparar datos
        X_train, y_train, X_test, y_test = trainer.prepare_data_for_training(test_size=0.2)
        
        # Entrenar modelo
        balance = input("⚖️ ¿Balancear clases automáticamente? [S/n]: ").strip().lower()
        balance_classes = balance != 'n'
        
        model = trainer.train_with_cross_validation(cv_folds=3, balance_classes=balance_classes)
        
        # Evaluar
        accuracy, f1 = trainer.evaluate_final_performance()
        
        # Mostrar características importantes
        trainer.get_feature_importance(top_n=15)
        
        # Generar visualizaciones
        trainer.plot_results()
        
        # Guardar resultados
        save_results = input("\n💾 ¿Guardar resultados? [S/n]: ").strip().lower()
        if save_results != 'n':
            output_dir = input("📁 Directorio de salida [./resultados_mlp]: ").strip()
            if not output_dir:
                output_dir = "./resultados_mlp"
            trainer.save_results(output_dir)
        
        print(f"\n🎉 ENTRENAMIENTO COMPLETADO!")
        print(f"   Videos procesados: {len(video_features)}")
        print(f"   Videos entrenamiento: {len(X_train)}")
        print(f"   Videos test: {len(X_test)}")
        print(f"   Accuracy final: {accuracy:.4f}")
        print(f"   F1-Score final: {f1:.4f}")
        
        # Resumen por clase
        print(f"\n📊 Distribución de clases:")
        class_counts = video_features['clase'].value_counts()
        for clase, count in class_counts.items():
            percentage = (count / len(video_features)) * 100
            print(f"   {clase}: {count} videos ({percentage:.1f}%)")
        
    except KeyboardInterrupt:
        print(f"\n⏹️ Entrenamiento interrumpido por el usuario")
    except Exception as e:
        print(f"\n❌ Error durante el entrenamiento: {e}")
        import traceback
        traceback.print_exc()

class VideoPredictor:
    """
    Clase para hacer predicciones en nuevos videos usando el modelo entrenado.
    """
    
    def __init__(self, model_path):
        """
        Carga un modelo entrenado.
        
        Parámetros:
        -----------
        model_path : str
            Ruta al archivo del modelo (.pkl)
        """
        import pickle
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.feature_names = model_data['feature_names']
        
        print(f"✅ Modelo cargado desde: {model_path}")
        print(f"   Clases: {list(self.label_encoder.classes_)}")
        print(f"   Características: {len(self.feature_names)}")
    
    def predict_video(self, csv_path, return_probabilities=False):
        """
        Predice la clase de un nuevo video.
        
        Parámetros:
        -----------
        csv_path : str
            Ruta al archivo CSV del video
        return_probabilities : bool
            Si retornar probabilidades además de la predicción
        
        Retorna:
        --------
        str o tuple
            Clase predicha (y probabilidades si se solicitan)
        """
        # Crear un trainer temporal solo para extraer características
        temp_trainer = VideoLevelMLPTrainer()
        
        # Extraer características del video (clase dummy)
        features = temp_trainer.extract_video_features(csv_path, 'dummy')
        
        if features is None:
            raise ValueError(f"No se pudieron extraer características de {csv_path}")
        
        # Convertir a DataFrame
        features_df = pd.DataFrame([features])
        
        # Seleccionar solo las características que el modelo conoce
        available_features = [f for f in self.feature_names if f in features_df.columns]
        missing_features = [f for f in self.feature_names if f not in features_df.columns]
        
        if missing_features:
            print(f"⚠️ Características faltantes: {len(missing_features)}")
            # Crear DataFrame con todas las características, rellenando con 0 las faltantes
            full_features_df = pd.DataFrame(columns=self.feature_names)
            for feature in self.feature_names:
                if feature in features_df.columns:
                    full_features_df[feature] = features_df[feature]
                else:
                    full_features_df[feature] = 0
            features_df = full_features_df
        else:
            features_df = features_df[self.feature_names]
        
        # Rellenar NaN
        features_df = features_df.fillna(0)
        
        # Escalar
        features_scaled = self.scaler.transform(features_df)
        
        # Predecir
        prediction_encoded = self.model.predict(features_scaled)[0]
        prediction_class = self.label_encoder.inverse_transform([prediction_encoded])[0]
        
        if return_probabilities:
            probabilities = self.model.predict_proba(features_scaled)[0]
            class_probs = dict(zip(self.label_encoder.classes_, probabilities))
            return prediction_class, class_probs
        else:
            return prediction_class

def predict_new_video():
    """
    Función auxiliar para predecir un nuevo video.
    """
    print("🔮 PREDICCIÓN DE NUEVO VIDEO")
    print("=" * 40)
    
    model_path = input("📁 Ruta al modelo entrenado (.pkl): ").strip()
    if not model_path or not Path(model_path).exists():
        print("❌ Modelo no encontrado")
        return
    
    video_path = input("🎬 Ruta al video CSV: ").strip()
    if not video_path or not Path(video_path).exists():
        print("❌ Video no encontrado")
        return
    
    try:
        predictor = VideoPredictor(model_path)
        prediction, probabilities = predictor.predict_video(video_path, return_probabilities=True)
        
        print(f"\n🎯 PREDICCIÓN:")
        print(f"   Clase predicha: {prediction}")
        print(f"\n📊 Probabilidades:")
        for clase, prob in probabilities.items():
            print(f"   {clase}: {prob:.4f} ({prob*100:.1f}%)")
        
    except Exception as e:
        print(f"❌ Error en predicción: {e}")

if __name__ == "__main__":
    print("Seleccione una opción:")
    print("1. Entrenar nuevo modelo")
    print("2. Predecir con modelo existente")
    
    choice = input("Opción [1]: ").strip()
    
    if choice == "2":
        predict_new_video()
    else:
        main()