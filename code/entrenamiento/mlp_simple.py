"""
Entrenador MLP simplificado que usa solo características básicas originales.
No genera estadísticas complejas, solo usa las más importantes.
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
import warnings
warnings.filterwarnings('ignore')

class SimpleVideoMLPTrainer:
    """
    Entrenador MLP simplificado usando solo características básicas.
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model = None
        self.video_features = None
        self.results = {}
    
    def extract_simple_features(self, csv_path, video_class):
        """
        Extrae solo las características más importantes y simples.
        """
        try:
            df = pd.read_csv(csv_path)
            
            if df.empty:
                return None
            
            video_name = Path(csv_path).stem
            features = {'video_id': video_name, 'clase': video_class}
            
            # 1. CARACTERÍSTICAS BÁSICAS MÁS IMPORTANTES
            # Promedios simples de las columnas numéricas principales
            important_cols = [
                'Velocidad', 'Aceleracion', 'Direccion', 'Densidad',
                'Linealidad', 'Circularidad', 'Zigzag'
            ]
            
            for col in important_cols:
                if col in df.columns and not df[col].isna().all():
                    data = df[col].dropna()
                    if len(data) > 0:
                        features[f'{col}_promedio'] = data.mean()
                        features[f'{col}_maximo'] = data.max()
            
            # 2. CARACTERÍSTICAS DE COMPORTAMIENTO SIMPLES
            # Duracion del video
            features['duracion_frames'] = len(df)
            
            # Número de objetos únicos
            if 'Objeto' in df.columns:
                features['num_objetos'] = df['Objeto'].nunique()
            else:
                features['num_objetos'] = 1
            
            # Porcentaje de tiempo en interacción
            if 'En_Interaccion' in df.columns:
                features['pct_interaccion'] = (df['En_Interaccion'] == 1).mean()
            else:
                features['pct_interaccion'] = 0
            
            # Porcentaje de movimiento cíclico
            if 'Es_Ciclico' in df.columns:
                features['pct_ciclico'] = (df['Es_Ciclico'] == 1).mean()
            else:
                features['pct_ciclico'] = 0
            
            # 3. CARACTERÍSTICAS ESPACIALES BÁSICAS
            if 'Centroide_X' in df.columns and 'Centroide_Y' in df.columns:
                # Rango de movimiento
                features['rango_movimiento_x'] = df['Centroide_X'].max() - df['Centroide_X'].min()
                features['rango_movimiento_y'] = df['Centroide_Y'].max() - df['Centroide_Y'].min()
                
                # Centro promedio de actividad
                features['centro_actividad_x'] = df['Centroide_X'].mean()
                features['centro_actividad_y'] = df['Centroide_Y'].mean()
            
            # 4. CARACTERÍSTICAS DE FRECUENCIA SIMPLE
            if 'Frecuencia_Ciclo' in df.columns:
                freq_data = df['Frecuencia_Ciclo'].dropna()
                if len(freq_data) > 0:
                    features['frecuencia_promedio'] = freq_data.mean()
            
            if 'Area_Trayectoria' in df.columns:
                area_data = df['Area_Trayectoria'].dropna()
                if len(area_data) > 0:
                    features['area_promedio'] = area_data.mean()
            
            return features
            
        except Exception as e:
            print(f"Error procesando {csv_path}: {e}")
            return None
    
    def load_videos_from_folders(self, data_dir):
        """
        Carga videos y extrae características simples.
        """
        print("📁 Cargando videos (características simples)...")
        
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
                features = self.extract_simple_features(csv_file, class_name)
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
        
        print(f"\n📊 Resumen:")
        print(f"   Videos procesados: {len(self.video_features)}")
        print(f"   Características: {len(numeric_columns)}")
        print(f"   Clases: {dict(self.video_features['clase'].value_counts())}")
        
        return self.video_features
    
    def prepare_data(self, test_size=0.2):
        """
        Prepara datos para entrenamiento.
        """
        print(f"\n🔄 Preparando datos...")
        
        # Separar características y objetivo
        feature_cols = [col for col in self.video_features.columns 
                       if col not in ['video_id', 'clase']]
        
        X = self.video_features[feature_cols]
        y = self.video_features['clase']
        
        # Codificar clases
        y_encoded = self.label_encoder.fit_transform(y)
        
        print(f"   Características seleccionadas: {len(feature_cols)}")
        print(f"   {feature_cols}")
        
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
        
        # Escalar
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Guardar
        self.X_train = X_train_scaled
        self.y_train = y_train
        self.X_test = X_test_scaled
        self.y_test = y_test
        self.feature_names = feature_cols
        
        return X_train_scaled, y_train, X_test_scaled, y_test
    
    def train_simple_mlp(self, cv_folds=3):
        """
        Entrena MLP con configuraciones simples y efectivas.
        """
        print(f"\n🧠 Entrenando MLP simple...")
        
        n_samples = len(self.X_train)
        print(f"   Muestras entrenamiento: {n_samples}")
        
        # Configuraciones simples pero efectivas
        configs = [
            # Configuración muy simple
            {
                'hidden_layer_sizes': (8,),
                'activation': 'relu',
                'alpha': 0.1,
                'learning_rate': 'adaptive',
                'max_iter': 2000,
                'early_stopping': True,
                'random_state': self.random_state
            },
            # Configuración conservadora
            {
                'hidden_layer_sizes': (16,),
                'activation': 'tanh',
                'alpha': 0.05,
                'learning_rate': 'adaptive',
                'max_iter': 2000,
                'early_stopping': True,
                'random_state': self.random_state
            },
            # Configuración con dos capas
            {
                'hidden_layer_sizes': (12, 6),
                'activation': 'relu',
                'alpha': 0.01,
                'learning_rate': 'adaptive',
                'max_iter': 2000,
                'early_stopping': True,
                'random_state': self.random_state
            }
        ]
        
        # Ajustar CV
        cv_folds = min(cv_folds, n_samples // 5, 5)
        cv_folds = max(cv_folds, 2)
        
        best_score = -1
        best_config = None
        cv_results = []
        
        print(f"   Probando {len(configs)} configuraciones...")
        
        for i, config in enumerate(configs):
            try:
                mlp = MLPClassifier(**config)
                
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
                scores = cross_val_score(mlp, self.X_train, self.y_train, 
                                       cv=cv, scoring='f1_weighted')
                
                mean_score = scores.mean()
                
                cv_results.append({
                    'config': i + 1,
                    'architecture': str(config['hidden_layer_sizes']),
                    'mean_f1': mean_score,
                    'std_f1': scores.std()
                })
                
                print(f"   Config {i+1}: {config['hidden_layer_sizes']} → F1={mean_score:.4f}")
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_config = config
                
            except Exception as e:
                print(f"   Config {i+1}: ERROR - {e}")
        
        if best_config is None:
            raise ValueError("No se entrenó ningún modelo")
        
        print(f"\n🏆 Mejor configuración:")
        print(f"   Arquitectura: {best_config['hidden_layer_sizes']}")
        print(f"   F1-Score CV: {best_score:.4f}")
        
        # Entrenar modelo final
        self.model = MLPClassifier(**best_config)
        self.model.fit(self.X_train, self.y_train)
        
        self.results = {
            'best_config': best_config,
            'best_cv_score': best_score,
            'cv_results': cv_results
        }
        
        return self.model
    
    def evaluate_final(self):
        """
        Evaluación final en test.
        """
        print(f"\n🎯 EVALUACIÓN FINAL...")
        
        y_pred = self.model.predict(self.X_test)
        
        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        
        print(f"   Videos test: {len(self.y_test)}")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   F1-Score: {f1:.4f}")
        
        # Reporte por clase
        class_names = self.label_encoder.classes_
        report = classification_report(self.y_test, y_pred, target_names=class_names)
        print(f"\n📋 Reporte detallado:")
        print(report)
        
        # Matriz de confusión
        cm = confusion_matrix(self.y_test, y_pred)
        
        # Visualizar
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Matriz de Confusión')
        plt.ylabel('Real')
        plt.xlabel('Predicho')
        plt.show()
        
        self.results.update({
            'final_accuracy': accuracy,
            'final_f1': f1,
            'confusion_matrix': cm
        })
        
        return accuracy, f1
    
    def show_feature_importance(self):
        """
        Muestra características más importantes.
        """
        if self.model is None or not hasattr(self.model, 'coefs_'):
            return
        
        # Aproximación de importancia
        weights = np.abs(self.model.coefs_[0])
        importance = np.mean(weights, axis=1)
        
        # DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names[:len(importance)],
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        print(f"\n🔍 Características más importantes:")
        for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
            print(f"   {i+1:2d}. {row['feature']:<20}: {row['importance']:.4f}")
        
        return importance_df

def main():
    """
    Función principal simplificada.
    """
    print("🧠 ENTRENADOR MLP SIMPLE")
    print("=" * 40)
    
    data_dir = input("📁 Directorio con carpetas de clases: ").strip()
    if not data_dir or not Path(data_dir).exists():
        print("❌ Directorio no válido")
        return
    
    try:
        # Crear entrenador
        trainer = SimpleVideoMLPTrainer(random_state=42)
        
        # Cargar videos (características simples)
        videos = trainer.load_videos_from_folders(data_dir)
        
        # Preparar datos
        trainer.prepare_data(test_size=0.2)
        
        # Entrenar
        model = trainer.train_simple_mlp(cv_folds=3)
        
        # Evaluar
        accuracy, f1 = trainer.evaluate_final()
        
        # Mostrar características importantes
        trainer.show_feature_importance()
        
        print(f"\n🎉 COMPLETADO!")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   F1-Score: {f1:.4f}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()