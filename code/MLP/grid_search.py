"""
Script para realizar fine-tuning automático del perceptrón multicapa
generando combinaciones óptimas de hiperparámetros
"""
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

def cargar_y_combinar_csvs(directorio_csv):
    """Carga todos los archivos CSV del directorio y los combina"""
    import glob
    
    print(f"Buscando archivos CSV en {directorio_csv}...")
    csv_files = glob.glob(os.path.join(directorio_csv, "*.csv"))
    
    if not csv_files:
        print(f"No se encontraron archivos CSV en {directorio_csv}")
        return None
    
    print(f"Cargando {len(csv_files)} archivos CSV...")
    
    dataframes = []
    for csv_file in csv_files:
        nombre_video = os.path.basename(csv_file).split('.')[0]
        try:
            df = pd.read_csv(csv_file)
            df['Video'] = nombre_video  # Añadir columna de identificación del video
            dataframes.append(df)
            print(f"  ✓ Cargado {nombre_video}: {len(df)} registros")
        except Exception as e:
            print(f"  ✗ Error al cargar {nombre_video}: {str(e)}")
    
    if not dataframes:
        return None
    
    # Combinar todos los dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)
    print(f"Total de registros combinados: {len(combined_df)}")
    
    return combined_df

def limpiar_datos(df):
    """Limpia y preprocesa los datos"""
    # Reemplazar valores infinitos o NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Imputar valores faltantes para columnas numéricas
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        df[col].fillna(df[col].median(), inplace=True)
    
    # Imputar valores categóricos
    for col in df.select_dtypes(include=['object']).columns:
        df[col].fillna(df[col].mode()[0], inplace=True)
    
    return df

def preparar_datos(df, target_column, features_to_exclude=['Frame', 'Objeto', 'Video'], 
                  test_size=0.2, random_state=42, apply_pca=False, n_components=0.95):
    """Prepara los datos para entrenamiento"""
    # Seleccionar características y target
    features = [col for col in df.columns 
               if col != target_column 
               and col not in features_to_exclude]
    
    X = df[features]
    y = df[target_column]
    
    # Codificar la variable objetivo si es categórica
    le = None
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
        print(f"Clases codificadas: {dict(zip(le.classes_, range(len(le.classes_))))}")
    
    # División train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, 
        stratify=y if len(np.unique(y)) > 1 else None
    )
    
    # Escalar características
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Aplicar PCA si está habilitado
    pca = None
    if apply_pca:
        print(f"Aplicando PCA para reducir dimensionalidad...")
        pca = PCA(n_components=n_components, random_state=random_state)
        X_train_scaled = pca.fit_transform(X_train_scaled)
        X_test_scaled = pca.transform(X_test_scaled)
        
        if isinstance(n_components, float):
            print(f"Varianza explicada: {sum(pca.explained_variance_ratio_):.2f}")
        else:
            print(f"Componentes seleccionados: {pca.n_components_}")
            print(f"Varianza explicada: {sum(pca.explained_variance_ratio_):.2f}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, pca, le

def realizar_grid_search(X_train, y_train, cv=5, n_jobs=-1):
    """
    Realiza Grid Search para encontrar los mejores hiperparámetros
    """
    print("Iniciando Grid Search para encontrar mejores hiperparámetros...")
    
    # Definir grid de parámetros a probar
    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (100, 100), (50, 30, 10)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam', 'sgd'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'adaptive'],
        'batch_size': ['auto', 64, 128],
        'max_iter': [200, 500]
    }
    
    # Inicializar modelo base
    mlp = MLPClassifier(random_state=42)
    
    # Configurar Grid Search
    grid = GridSearchCV(
        estimator=mlp,
        param_grid=param_grid,
        cv=cv,
        n_jobs=n_jobs,
        scoring='accuracy',
        verbose=2,
        return_train_score=True
    )
    
    # Realizar búsqueda
    print("Entrenando modelos, esto puede llevar tiempo...")
    grid.fit(X_train, y_train)
    
    # Obtener mejores resultados
    print(f"Mejor puntuación: {grid.best_score_:.4f}")
    print(f"Mejores parámetros: {grid.best_params_}")
    
    return grid

def realizar_random_search(X_train, y_train, n_iter=30, cv=5, n_jobs=-1):
    """
    Realiza Random Search para encontrar hiperparámetros eficientemente
    """
    print("Iniciando Random Search para encontrar mejores hiperparámetros...")
    
    # Definir grid de parámetros a probar
    param_distributions = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (100, 100), 
                               (50, 30, 10), (100, 50, 25), (50, 50, 50)],
        'activation': ['relu', 'tanh', 'logistic'],
        'solver': ['adam', 'sgd'],
        'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1],
        'learning_rate': ['constant', 'adaptive', 'invscaling'],
        'batch_size': ['auto', 32, 64, 128, 256],
        'max_iter': [200, 500, 1000]
    }
    
    # Inicializar modelo base
    mlp = MLPClassifier(random_state=42)
    
    # Configurar Random Search
    random_search = RandomizedSearchCV(
        estimator=mlp,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        n_jobs=n_jobs,
        scoring='accuracy',
        verbose=2,
        random_state=42,
        return_train_score=True
    )
    
    # Realizar búsqueda
    print("Entrenando modelos, esto puede llevar tiempo...")
    random_search.fit(X_train, y_train)
    
    # Obtener mejores resultados
    print(f"Mejor puntuación: {random_search.best_score_:.4f}")
    print(f"Mejores parámetros: {random_search.best_params_}")
    
    return random_search

def evaluar_modelo(model, X_test, y_test, label_encoder=None):
    """
    Evalúa el modelo final con el conjunto de prueba
    """
    print("\nEvaluando modelo en conjunto de prueba...")
    
    # Realizar predicciones
    y_pred = model.predict(X_test)
    
    # Calcular métricas básicas
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Precisión global: {accuracy:.4f}")
    
    # Generar informe detallado
    if label_encoder is not None and hasattr(label_encoder, 'classes_'):
        target_names = label_encoder.classes_
        print("\nInforme de clasificación:")
        print(classification_report(y_test, y_pred, target_names=target_names))
        
        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=target_names, yticklabels=target_names)
        plt.ylabel('Valor Real')
        plt.xlabel('Valor Predicho')
        plt.title('Matriz de Confusión')
        plt.tight_layout()
    else:
        print("\nInforme de clasificación:")
        print(classification_report(y_test, y_pred))
        
        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d')
        plt.ylabel('Valor Real')
        plt.xlabel('Valor Predicho')
        plt.title('Matriz de Confusión')
        plt.tight_layout()
    
    return accuracy, y_pred