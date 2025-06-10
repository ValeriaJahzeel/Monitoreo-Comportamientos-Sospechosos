"""
MLP Mejorado que extrae características más discriminativas y útiles.
Diseñado para mejorar el rendimiento cuando el MLP simple da baja precisión.
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
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

class ImprovedVideoMLPTrainer:
    """
    MLP mejorado que extrae características más discriminativas.
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_selector = SelectKBest(f_classif, k=15)
        self.model = None
        self.video_features = None
        self.results = {}
    
    def extract_comprehensive_features(self, csv_path, video_class):
        """
        Extrae características más completas y discriminativas.
        """
        try:
            df = pd.read_csv(csv_path)
            
            if df.empty:
                return None
            
            video_name = Path(csv_path).stem
            features = {'video_id': video_name, 'clase': video_class}
            
            # === CARACTERÍSTICAS NUMÉRICAS BÁSICAS ===
            numeric_cols = ['Velocidad', 'Aceleracion', 'Direccion', 'Densidad', 
                          'Linealidad', 'Circularidad', 'Zigzag', 'Frecuencia_Ciclo', 
                          'Amplitud_Ciclo', 'Area_Trayectoria', 'Desplazamiento']
            
            for col in numeric_cols:
                if col in df.columns and not df[col].isna().all():
                    data = df[col].dropna()
                    if len(data) > 0:
                        # Estadísticas básicas
                        features[f'{col}_mean'] = data.mean()
                        features[f'{col}_max'] = data.max()
                        features[f'{col}_min'] = data.min()
                        features[f'{col}_std'] = data.std() if len(data) > 1 else 0
                        features[f'{col}_median'] = data.median()
                        
                        # Percentiles para capturar distribución
                        features[f'{col}_p25'] = data.quantile(0.25)
                        features[f'{col}_p75'] = data.quantile(0.75)
                        
                        # Rango y coeficiente de variación
                        features[f'{col}_range'] = data.max() - data.min()
                        if data.mean() != 0:
                            features[f'{col}_cv'] = data.std() / abs(data.mean())
                        else:
                            features[f'{col}_cv'] = 0
            
            # === CARACTERÍSTICAS TEMPORALES AVANZADAS ===
            features['total_frames'] = len(df)
            features['frames_unicos'] = df['Frame'].nunique() if 'Frame' in df.columns else len(df)
            features['objetos_unicos'] = df['Objeto'].nunique() if 'Objeto' in df.columns else 1
            features['registros_por_frame'] = len(df) / features['frames_unicos'] if features['frames_unicos'] > 0 else 0
            
            # === CARACTERÍSTICAS DE COMPORTAMIENTO ===
            # Interacciones
            if 'En_Interaccion' in df.columns:
                features['pct_interaccion'] = (df['En_Interaccion'] == 1).mean()
                features['total_interacciones'] = (df['En_Interaccion'] == 1).sum()
                features['intensidad_interaccion'] = features['total_interacciones'] / features['total_frames']
            else:
                features['pct_interaccion'] = 0
                features['total_interacciones'] = 0
                features['intensidad_interaccion'] = 0
            
            # Patrones cíclicos
            if 'Es_Ciclico' in df.columns:
                features['pct_ciclico'] = (df['Es_Ciclico'] == 1).mean()
                features['total_ciclico'] = (df['Es_Ciclico'] == 1).sum()
                features['intensidad_ciclico'] = features['total_ciclico'] / features['total_frames']
            else:
                features['pct_ciclico'] = 0
                features['total_ciclico'] = 0
                features['intensidad_ciclico'] = 0
            
            # === CARACTERÍSTICAS DE POSTURA Y MOVIMIENTO ===
            if 'Postura' in df.columns:
                posturas = df['Postura'].value_counts(normalize=True)
                features['num_posturas_diferentes'] = len(posturas)
                features['postura_dominante_freq'] = posturas.iloc[0] if len(posturas) > 0 else 0
                features['diversidad_posturas'] = 1 - features['postura_dominante_freq']
                
                # Frecuencias de posturas específicas (si existen)
                for postura in posturas.head(3).index:  # Top 3 posturas
                    features[f'postura_{postura}_freq'] = posturas[postura]
            
            if 'Patron_Movimiento' in df.columns:
                patrones = df['Patron_Movimiento'].value_counts(normalize=True)
                features['num_patrones_diferentes'] = len(patrones)
                features['patron_dominante_freq'] = patrones.iloc[0] if len(patrones) > 0 else 0
                features['diversidad_patrones'] = 1 - features['patron_dominante_freq']
                
                # Frecuencias de patrones específicos
                for patron in patrones.head(3).index:  # Top 3 patrones
                    features[f'patron_{patron}_freq'] = patrones[patron]
            
            # === CARACTERÍSTICAS ESPACIALES ===
            if 'Centroide_X' in df.columns and 'Centroide_Y' in df.columns:
                # Rangos de movimiento
                features['rango_x'] = df['Centroide_X'].max() - df['Centroide_X'].min()
                features['rango_y'] = df['Centroide_Y'].max() - df['Centroide_Y'].min()
                features['area_total_movimiento'] = features['rango_x'] * features['rango_y']
                
                # Centros de actividad
                features['centro_x'] = df['Centroide_X'].mean()
                features['centro_y'] = df['Centroide_Y'].mean()
                
                # Variabilidad espacial
                features['std_x'] = df['Centroide_X'].std()
                features['std_y'] = df['Centroide_Y'].std()
                
                # Distancia promedio del centro
                center_x, center_y = features['centro_x'], features['centro_y']
                distances = np.sqrt((df['Centroide_X'] - center_x)**2 + (df['Centroide_Y'] - center_y)**2)
                features['distancia_promedio_centro'] = distances.mean()
                features['distancia_max_centro'] = distances.max()
            
            # === CARACTERÍSTICAS DE CAMBIOS Y TRANSICIONES ===
            # Cambios en velocidad
            if 'Velocidad' in df.columns:
                velocidades = df['Velocidad'].dropna()
                if len(velocidades) > 1:
                    cambios_vel = np.abs(np.diff(velocidades))
                    features['cambios_velocidad_mean'] = cambios_vel.mean()
                    features['cambios_velocidad_max'] = cambios_vel.max()
                    features['cambios_velocidad_std'] = cambios_vel.std()
                    
                    # Aceleraciones y frenadas bruscas
                    features['aceleraciones_bruscas'] = (cambios_vel > cambios_vel.mean() + 2*cambios_vel.std()).sum()
                    features['periodos_quieto'] = (velocidades == 0).sum()
                    features['pct_tiempo_quieto'] = features['periodos_quieto'] / len(velocidades)
            
            # Cambios en dirección
            if 'Direccion' in df.columns:
                direcciones = df['Direccion'].dropna()
                if len(direcciones) > 1:
                    # Calcular cambios de dirección (considerando circularidad)
                    diff_dir = np.diff(direcciones)
                    diff_dir = np.minimum(np.abs(diff_dir), 360 - np.abs(diff_dir))  # Distancia circular
                    
                    features['cambios_direccion_mean'] = diff_dir.mean()
                    features['cambios_direccion_max'] = diff_dir.max()
                    features['cambios_direccion_std'] = diff_dir.std()
                    
                    # Giros bruscos (cambios > 90 grados)
                    features['giros_bruscos'] = (diff_dir > 90).sum()
                    features['pct_giros_bruscos'] = features['giros_bruscos'] / len(diff_dir)
            
            # === CARACTERÍSTICAS DE DENSIDAD Y CONCENTRACIÓN ===
            if 'Densidad' in df.columns:
                densidades = df['Densidad'].dropna()
                if len(densidades) > 0:
                    # Variaciones de densidad
                    features['densidad_variabilidad'] = densidades.std()
                    features['densidad_range'] = densidades.max() - densidades.min()
                    
                    # Períodos de alta/baja densidad
                    umbral_alto = densidades.quantile(0.75)
                    umbral_bajo = densidades.quantile(0.25)
                    features['periodos_alta_densidad'] = (densidades > umbral_alto).sum()
                    features['periodos_baja_densidad'] = (densidades < umbral_bajo).sum()
            
            # === CARACTERÍSTICAS ESPECÍFICAS DE COMPORTAMIENTO ===
            # Patrones de zigzag (posible indicador de merodeo)
            if 'Zigzag' in df.columns:
                zigzag_data = df['Zigzag'].dropna()
                if len(zigzag_data) > 0:
                    features['zigzag_intensidad'] = zigzag_data.mean()
                    features['zigzag_picos'] = (zigzag_data > zigzag_data.quantile(0.8)).sum()
            
            # Linealidad vs circularidad (patrones de movimiento)
            if 'Linealidad' in df.columns and 'Circularidad' in df.columns:
                lineal = df['Linealidad'].dropna()
                circular = df['Circularidad'].dropna()
                if len(lineal) > 0 and len(circular) > 0:
                    features['ratio_lineal_circular'] = lineal.mean() / (circular.mean() + 1e-6)
                    features['preferencia_movimiento'] = 'lineal' if features['ratio_lineal_circular'] > 1 else 'circular'
                    # Convertir a numérico
                    features['preferencia_movimiento_num'] = 1 if features['ratio_lineal_circular'] > 1 else 0
                    del features['preferencia_movimiento']  # Eliminar la versión categórica
            
            return features
            
        except Exception as e:
            print(f"Error procesando {csv_path}: {e}")
            return None
    
    def load_videos_with_rich_features(self, data_dir):
        """
        Carga videos y extrae características completas.
        """
        print("📁 Cargando videos con características mejoradas...")
        
        data_dir = Path(data_dir)
        classes = ['normal', 'merodeo', 'forcejeo']
        
        all_features = []
        
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
                features = self.extract_comprehensive_features(csv_file, class_name)
                if features is not None:
                    all_features.append(features)
                    print(f"   ✅ {csv_file.name}")
                else:
                    print(f"   ❌ {csv_file.name}")
        
        if not all_features:
            raise ValueError("No se procesaron videos")
        
        # Convertir a DataFrame
        self.video_features = pd.DataFrame(all_features)
        
        # Rellenar NaN con 0
        numeric_columns = self.video_features.select_dtypes(include=[np.number]).columns
        self.video_features[numeric_columns] = self.video_features[numeric_columns].fillna(0)
        
        # Reemplazar infinitos
        self.video_features[numeric_columns] = self.video_features[numeric_columns].replace([np.inf, -np.inf], 0)
        
        print(f"\n📊 Características extraídas:")
        print(f"   Videos procesados: {len(self.video_features)}")
        print(f"   Características totales: {len(numeric_columns)}")
        print(f"   Distribución de clases: {dict(self.video_features['clase'].value_counts())}")
        
        return self.video_features
    
    def prepare_data_with_feature_selection(self, test_size=0.2, n_features=15):
        """
        Prepara datos con selección automática de características.
        """
        print(f"\n🔄 Preparando datos con selección de características...")
        
        # Separar características y objetivo
        feature_cols = [col for col in self.video_features.columns 
                       if col not in ['video_id', 'clase']]
        
        X = self.video_features[feature_cols]
        y = self.video_features['clase']
        
        print(f"   Características disponibles: {len(feature_cols)}")
        
        # Codificar clases
        y_encoded = self.label_encoder.fit_transform(y)
        
        # División estratificada
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=test_size, 
                stratify=y_encoded, random_state=self.random_state
            )
            print(f"   ✅ División estratificada")
        except:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=test_size, random_state=self.random_state
            )
            print(f"   ⚠️ División simple")
        
        print(f"   Train: {len(X_train)} | Test: {len(X_test)}")
        
        # Escalar primero
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Selección de características usando solo datos de entrenamiento
        print(f"   🎯 Seleccionando {n_features} mejores características...")
        
        self.feature_selector.k = min(n_features, len(feature_cols))
        X_train_selected = self.feature_selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = self.feature_selector.transform(X_test_scaled)
        
        # Obtener nombres de características seleccionadas
        selected_indices = self.feature_selector.get_support(indices=True)
        selected_features = [feature_cols[i] for i in selected_indices]
        
        print(f"   ✅ Características seleccionadas:")
        for i, feature in enumerate(selected_features[:10]):  # Mostrar top 10
            score = self.feature_selector.scores_[selected_indices[i]]
            print(f"      {i+1}. {feature}: {score:.2f}")
        
        # Guardar datos
        self.X_train = X_train_selected
        self.y_train = y_train
        self.X_test = X_test_selected
        self.y_test = y_test
        self.selected_features = selected_features
        
        return X_train_selected, y_train, X_test_selected, y_test
    
    def train_improved_mlp(self, cv_folds=3):
        """
        Entrena MLP mejorado con mejores configuraciones.
        """
        print(f"\n🧠 Entrenando MLP mejorado...")
        
        n_samples = len(self.X_train)
        n_features = self.X_train.shape[1]
        
        print(f"   Muestras entrenamiento: {n_samples}")
        print(f"   Características seleccionadas: {n_features}")
        
        # Configuraciones más robustas
        configs = [
            # Configuración conservadora
            {
                'hidden_layer_sizes': (10,),
                'activation': 'relu',
                'alpha': 0.01,
                'learning_rate': 'adaptive',
                'learning_rate_init': 0.001,
                'max_iter': 2000,
                'early_stopping': True,
                'validation_fraction': 0.2,
                'random_state': self.random_state
            },
            # Configuración con dos capas
            {
                'hidden_layer_sizes': (20, 10),
                'activation': 'relu',
                'alpha': 0.001,
                'learning_rate': 'adaptive',
                'learning_rate_init': 0.001,
                'max_iter': 2000,
                'early_stopping': True,
                'validation_fraction': 0.2,
                'random_state': self.random_state
            },
            # Configuración con activación tanh
            {
                'hidden_layer_sizes': (15,),
                'activation': 'tanh',
                'alpha': 0.01,
                'learning_rate': 'adaptive',
                'learning_rate_init': 0.01,
                'max_iter': 2000,
                'early_stopping': True,
                'validation_fraction': 0.2,
                'random_state': self.random_state
            },
            # Configuración más compleja
            {
                'hidden_layer_sizes': (25, 15, 8),
                'activation': 'relu',
                'alpha': 0.0001,
                'learning_rate': 'adaptive',
                'learning_rate_init': 0.0001,
                'max_iter': 3000,
                'early_stopping': True,
                'validation_fraction': 0.2,
                'random_state': self.random_state
            }
        ]
        
        # Ajustar CV
        cv_folds = min(cv_folds, n_samples // 5, 5)
        cv_folds = max(cv_folds, 2)
        
        best_score = -1
        best_config = None
        cv_results = []
        
        print(f"   Probando {len(configs)} configuraciones con {cv_folds}-fold CV...")
        
        for i, config in enumerate(configs):
            try:
                print(f"\n   🔄 Config {i+1}: {config['hidden_layer_sizes']}")
                
                mlp = MLPClassifier(**config)
                
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
                scores = cross_val_score(mlp, self.X_train, self.y_train, 
                                       cv=cv, scoring='f1_weighted', n_jobs=-1)
                
                mean_score = scores.mean()
                std_score = scores.std()
                
                cv_results.append({
                    'config': i + 1,
                    'architecture': str(config['hidden_layer_sizes']),
                    'activation': config['activation'],
                    'alpha': config['alpha'],
                    'mean_f1': mean_score,
                    'std_f1': std_score
                })
                
                print(f"      F1-Score: {mean_score:.4f} (±{std_score:.4f})")
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_config = config
                    print(f"      🏆 Nueva mejor configuración!")
                
            except Exception as e:
                print(f"      ❌ Error: {e}")
        
        if best_config is None:
            raise ValueError("No se entrenó ningún modelo")
        
        print(f"\n🏆 MEJOR CONFIGURACIÓN ENCONTRADA:")
        print(f"   Arquitectura: {best_config['hidden_layer_sizes']}")
        print(f"   Activación: {best_config['activation']}")
        print(f"   Alpha: {best_config['alpha']}")
        print(f"   F1-Score CV: {best_score:.4f}")
        
        # Entrenar modelo final
        self.model = MLPClassifier(**best_config)
        self.model.fit(self.X_train, self.y_train)
        
        self.results = {
            'best_config': best_config,
            'best_cv_score': best_score,
            'cv_results': cv_results,
            'selected_features': self.selected_features
        }
        
        return self.model
    
    def evaluate_final(self):
        """
        Evaluación final mejorada.
        """
        print(f"\n🎯 EVALUACIÓN FINAL...")
        
        y_pred = self.model.predict(self.X_test)
        y_pred_proba = self.model.predict_proba(self.X_test)
        
        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        
        print(f"   Videos test: {len(self.y_test)}")
        print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
        print(f"   F1-Score: {f1:.4f}")
        
        # Reporte por clase
        class_names = self.label_encoder.classes_
        report = classification_report(self.y_test, y_pred, target_names=class_names)
        print(f"\n📋 Reporte detallado:")
        print(report)
        
        # Matriz de confusión
        cm = confusion_matrix(self.y_test, y_pred)
        
        # Análisis de errores
        print(f"\n🔍 Análisis de predicciones:")
        for i, (true_label, pred_label, proba) in enumerate(zip(self.y_test, y_pred, y_pred_proba)):
            true_class = class_names[true_label]
            pred_class = class_names[pred_label]
            confidence = proba[pred_label]
            
            if true_label != pred_label:
                print(f"   ❌ Video {i}: {true_class} → {pred_class} (conf: {confidence:.3f})")
            else:
                print(f"   ✅ Video {i}: {true_class} (conf: {confidence:.3f})")
        
        # Visualizar matriz de confusión
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Matriz de Confusión - MLP Mejorado')
        plt.ylabel('Real')
        plt.xlabel('Predicho')
        plt.show()
        
        self.results.update({
            'final_accuracy': accuracy,
            'final_f1': f1,
            'confusion_matrix': cm
        })
        
        return accuracy, f1

def main():
    """
    Función principal para el MLP mejorado.
    """
    print("🧠 MLP MEJORADO CON CARACTERÍSTICAS AVANZADAS")
    print("=" * 55)
    
    data_dir = input("📁 Directorio con carpetas de clases: ").strip()
    if not data_dir or not Path(data_dir).exists():
        print("❌ Directorio no válido")
        return
    
    try:
        # Crear entrenador
        trainer = ImprovedVideoMLPTrainer(random_state=42)
        
        # Cargar videos con características avanzadas
        videos = trainer.load_videos_with_rich_features(data_dir)
        
        # Preparar datos con selección de características
        n_features = int(input("🎯 Número de características a seleccionar [15]: ") or "15")
        trainer.prepare_data_with_feature_selection(test_size=0.2, n_features=n_features)
        
        # Entrenar
        model = trainer.train_improved_mlp(cv_folds=3)
        
        # Evaluar
        accuracy, f1 = trainer.evaluate_final()
        
        print(f"\n🎉 ENTRENAMIENTO MLP MEJORADO COMPLETADO!")
        print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
        print(f"   F1-Score: {f1:.4f}")
        
        if accuracy > 0.7:
            print("   🎯 ¡Excelente resultado! El MLP mejorado funciona bien.")
        elif accuracy > 0.5:
            print("   ⚠️ Resultado moderado. Considera probar LSTM.")
        else:
            print("   ❌ Resultado bajo. Definitivamente prueba LSTM.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()