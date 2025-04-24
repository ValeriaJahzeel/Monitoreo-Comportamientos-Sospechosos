"""
Módulo de entrenamiento de perceptrón multicapa para análisis de comportamientos
Este script procesa archivos CSV de diferentes tamaños y entrena un modelo para clasificar
comportamientos normales vs. sospechosos.
"""
import os
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from concurrent.futures import ProcessPoolExecutor
import warnings
warnings.filterwarnings('ignore')

class MLPModelTrainer:
    def __init__(self, csv_directory, output_directory='./models', 
                 target_column='Patron_Movimiento', 
                 features_to_exclude=['Frame', 'Objeto', 'Video']):
        """
        Inicializa el entrenador del modelo MLP
        
        Parámetros:
        -----------
        csv_directory : str
            Ruta al directorio que contiene los archivos CSV con datos de movimiento
        output_directory : str
            Directorio donde se guardarán los modelos entrenados
        target_column : str
            Nombre de la columna objetivo para clasificación
        features_to_exclude : list
            Lista de columnas a excluir del entrenamiento
        """
        self.csv_directory = csv_directory
        self.output_directory = output_directory
        self.target_column = target_column
        self.features_to_exclude = features_to_exclude
        self.dataframes = {}  # Almacena DataFrames por archivo
        self.combined_data = None  # Para análisis combinado
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Crear directorio de salida si no existe
        os.makedirs(output_directory, exist_ok=True)
    
    def cargar_datos(self, max_workers=4):
        """
        Carga todos los archivos CSV del directorio especificado
        con soporte para procesamiento en paralelo
        """
        csv_files = glob.glob(os.path.join(self.csv_directory, "*.csv"))
        
        if not csv_files:
            print(f"No se encontraron archivos CSV en {self.csv_directory}")
            return False
        
        print(f"Cargando {len(csv_files)} archivos CSV...")
        
        # Cargar archivos en paralelo
        def load_file(file_path):
            try:
                filename = os.path.basename(file_path).split('.')[0]
                df = pd.read_csv(file_path)
                df['Video'] = filename  # Añadir columna de identificación del video
                return filename, df
            except Exception as e:
                print(f"  ✗ Error al cargar {file_path}: {str(e)}")
                return None, None
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(load_file, csv_files))
        
        # Filtrar resultados válidos
        valid_results = [(name, df) for name, df in results if name is not None]
        
        # Almacenar dataframes
        for name, df in valid_results:
            self.dataframes[name] = df
            print(f"  ✓ Cargado {name}: {len(df)} registros")
        
        # Crear DataFrame combinado para análisis global
        if self.dataframes:
            self.combined_data = pd.concat(self.dataframes.values(), ignore_index=True)
            print(f"Total de registros combinados: {len(self.combined_data)}")
            return True
        return False
    
    def limpiar_datos(self):
        """Limpia y preprocesa los datos para entrenamiento"""
        if self.combined_data is None:
            print("No hay datos cargados para limpiar")
            return
        
        print("Limpiando y preprocesando datos...")
        
        # Reemplazar valores infinitos o NaN
        self.combined_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Calcular estadísticas antes de la limpieza
        total_inicial = len(self.combined_data)
        nan_inicial = self.combined_data.isna().sum().sum()
        
        # Imputar valores faltantes con la mediana para columnas numéricas
        columnas_numericas = self.combined_data.select_dtypes(include=['float64', 'int64']).columns
        for col in columnas_numericas:
            self.combined_data[col].fillna(self.combined_data[col].median(), inplace=True)
        
        # Imputar valores categóricos con el valor más frecuente
        columnas_categoricas = self.combined_data.select_dtypes(include=['object']).columns
        for col in columnas_categoricas:
            self.combined_data[col].fillna(self.combined_data[col].mode()[0], inplace=True)
        
        # Verificar limpieza
        nan_final = self.combined_data.isna().sum().sum()
        print(f"  Valores NaN/Infinitos: {nan_inicial} → {nan_final}")
        print(f"  Registros totales: {total_inicial}")
    
    def preparar_datos_entrenamiento(self, test_size=0.2, random_state=42, apply_pca=False, n_components=0.95):
        """
        Prepara los datos para entrenamiento, con opción de aplicar PCA
        
        Parámetros:
        -----------
        test_size : float
            Proporción de datos para test (0-1)
        random_state : int
            Semilla para reproducibilidad
        apply_pca : bool
            Si es True, aplica PCA para reducción de dimensionalidad
        n_components : float o int
            Número de componentes o varianza a preservar
        """
        if self.combined_data is None:
            print("No hay datos cargados para preparar")
            return False
        
        print("Preparando datos para entrenamiento...")
        
        # Seleccionar características y variable objetivo
        features = [col for col in self.combined_data.columns 
                   if col != self.target_column 
                   and col not in self.features_to_exclude]
        
        X = self.combined_data[features]
        y = self.combined_data[self.target_column]
        
        # Codificar la variable objetivo si es categórica
        if y.dtype == 'object':
            y = self.label_encoder.fit_transform(y)
            print(f"  Clases codificadas: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")
        
        # Dividir en entrenamiento y prueba
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y if len(np.unique(y)) > 1 else None
        )
        
        # Escalar las características
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        # Aplicar PCA si está habilitado
        if apply_pca:
            print(f"  Aplicando PCA para reducir dimensionalidad...")
            pca = PCA(n_components=n_components, random_state=random_state)
            self.X_train = pca.fit_transform(self.X_train)
            self.X_test = pca.transform(self.X_test)
            
            if isinstance(n_components, float):
                print(f"  Varianza explicada: {sum(pca.explained_variance_ratio_):.2f}")
            else:
                print(f"  Componentes seleccionados: {pca.n_components_}")
                print(f"  Varianza explicada: {sum(pca.explained_variance_ratio_):.2f}")
            
            # Guardar el modelo PCA para usar en predicciones
            joblib.dump(pca, os.path.join(self.output_directory, 'pca_model.pkl'))
        
        print(f"  Datos preparados: X_train shape: {self.X_train.shape}, X_test shape: {self.X_test.shape}")
        return True
    
    def entrenar_modelo(self, hidden_layer_sizes=(100, 50), activation='relu', 
                        solver='adam', alpha=0.0001, batch_size='auto', 
                        learning_rate='adaptive', max_iter=200, random_state=42):
        """
        Entrena un modelo de Perceptrón Multicapa con los parámetros especificados
        """
        if self.X_train is None or self.y_train is None:
            print("Datos de entrenamiento no preparados")
            return False
        
        print("\nEntrenando Perceptrón Multicapa...")
        print(f"  Arquitectura: {hidden_layer_sizes}")
        print(f"  Activación: {activation}")
        print(f"  Solver: {solver}")
        print(f"  Alpha: {alpha}")
        print(f"  Batch size: {batch_size}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Max iteraciones: {max_iter}")
        
        # Inicializar y entrenar el modelo
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_iter=max_iter,
            random_state=random_state,
            verbose=True
        )
        
        # Entrenar modelo
        self.model.fit(self.X_train, self.y_train)
        
        # Evaluar en conjunto de prueba
        train_score = self.model.score(self.X_train, self.y_train)
        test_score = self.model.score(self.X_test, self.y_test)
        
        print(f"  Precisión en entrenamiento: {train_score:.4f}")
        print(f"  Precisión en prueba: {test_score:.4f}")
        
        return True
    
    def evaluar_modelo(self):
        """Evalúa el modelo entrenado y muestra métricas detalladas"""
        if self.model is None:
            print("No hay modelo entrenado para evaluar")
            return
        
        print("\nEvaluando modelo...")
        
        # Realizar predicciones
        y_pred = self.model.predict(self.X_test)
        
        # Calcular métricas
        accuracy = accuracy_score(self.y_test, y_pred)
        
        print(f"  Precisión global: {accuracy:.4f}")
        print("\nInforme de clasificación:")
        
        # Si la variable objetivo era categórica, decodificar para mejor interpretación
        if hasattr(self, 'label_encoder') and hasattr(self.label_encoder, 'classes_'):
            target_names = self.label_encoder.classes_
            print(classification_report(self.y_test, y_pred, target_names=target_names))
            
            # Matriz de confusión
            cm = confusion_matrix(self.y_test, y_pred)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', xticklabels=target_names, yticklabels=target_names)
            plt.ylabel('Valor Real')
            plt.xlabel('Valor Predicho')
            plt.title('Matriz de Confusión')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_directory, 'confusion_matrix.png'))
            print(f"  Matriz de confusión guardada en {os.path.join(self.output_directory, 'confusion_matrix.png')}")
        else:
            print(classification_report(self.y_test, y_pred))
            
            # Matriz de confusión
            cm = confusion_matrix(self.y_test, y_pred)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d')
            plt.ylabel('Valor Real')
            plt.xlabel('Valor Predicho')
            plt.title('Matriz de Confusión')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_directory, 'confusion_matrix.png'))
            print(f"  Matriz de confusión guardada en {os.path.join(self.output_directory, 'confusion_matrix.png')}")
    
    def optimizar_hiperparametros(self, method='grid', n_iter=20, cv=5, n_jobs=-1):
        """
        Optimiza hiperparámetros usando Grid Search o Random Search
        
        Parámetros:
        -----------
        method : str
            'grid' para GridSearchCV o 'random' para RandomizedSearchCV
        n_iter : int
            Número de iteraciones para Random Search
        cv : int
            Número de folds para validación cruzada
        n_jobs : int
            Número de trabajos en paralelo (-1 para usar todos los núcleos)
        """
        if self.X_train is None or self.y_train is None:
            print("Datos de entrenamiento no preparados")
            return None
        
        print(f"\nOptimizando hiperparámetros usando {method.capitalize()} Search...")
        
        # Definir grid de hiperparámetros
        param_grid = {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (100, 100), (50, 30, 10)],
            'activation': ['relu', 'tanh', 'logistic'],
            'solver': ['adam', 'sgd'],
            'alpha': [0.0001, 0.001, 0.01, 0.1],
            'learning_rate': ['constant', 'adaptive'],
            'batch_size': ['auto', 32, 64, 128],
            'max_iter': [200, 500, 1000]
        }
        
        # Inicializar el modelo base
        mlp = MLPClassifier(random_state=42)
        
        # Usar el método seleccionado
        if method.lower() == 'grid':
            search = GridSearchCV(
                mlp, param_grid, cv=cv, n_jobs=n_jobs, verbose=2, scoring='accuracy'
            )
        else:  # Random Search
            search = RandomizedSearchCV(
                mlp, param_grid, n_iter=n_iter, cv=cv, n_jobs=n_jobs, 
                verbose=2, random_state=42, scoring='accuracy'
            )
        
        # Realizar búsqueda
        search.fit(self.X_train, self.y_train)
        
        # Obtener mejores parámetros
        print("\nMejores hiperparámetros encontrados:")
        print(f"  {search.best_params_}")
        print(f"  Mejor puntuación: {search.best_score_:.4f}")
        
        # Actualizar el modelo con los mejores parámetros
        self.model = search.best_estimator_
        
        # Evaluar en conjunto de prueba
        test_score = self.model.score(self.X_test, self.y_test)
        print(f"  Precisión en prueba con mejores parámetros: {test_score:.4f}")
        
        # Guardar resultados de la búsqueda
        search_results = pd.DataFrame(search.cv_results_)
        search_results.to_csv(os.path.join(self.output_directory, f'{method}_search_results.csv'), index=False)
        print(f"  Resultados guardados en {os.path.join(self.output_directory, f'{method}_search_results.csv')}")
        
        return search.best_params_
    
    def guardar_modelo(self, nombre_modelo='mlp_model'):
        """Guarda el modelo entrenado y sus componentes asociados"""
        if self.model is None:
            print("No hay modelo para guardar")
            return
        
        print("\nGuardando modelo y componentes...")
        
        # Guardar modelo
        joblib.dump(self.model, os.path.join(self.output_directory, f'{nombre_modelo}.pkl'))
        print(f"  Modelo guardado en {os.path.join(self.output_directory, f'{nombre_modelo}.pkl')}")
        
        # Guardar scaler
        joblib.dump(self.scaler, os.path.join(self.output_directory, 'scaler.pkl'))
        print(f"  Scaler guardado en {os.path.join(self.output_directory, 'scaler.pkl')}")
        
        # Guardar label encoder si existe
        if hasattr(self, 'label_encoder') and hasattr(self.label_encoder, 'classes_'):
            joblib.dump(self.label_encoder, os.path.join(self.output_directory, 'label_encoder.pkl'))
            print(f"  Label Encoder guardado en {os.path.join(self.output_directory, 'label_encoder.pkl')}")
        
        # Guardar una descripción del modelo
        with open(os.path.join(self.output_directory, f'{nombre_modelo}_info.txt'), 'w') as f:
            f.write(f"Arquitectura: {self.model.hidden_layer_sizes}\n")
            f.write(f"Activación: {self.model.activation}\n")
            f.write(f"Solver: {self.model.solver}\n")
            f.write(f"Alpha: {self.model.alpha}\n")
            f.write(f"Learning rate: {self.model.learning_rate}\n")
            f.write(f"Batch size: {self.model.batch_size}\n")
            f.write(f"Número de capas: {self.model.n_layers_}\n")
            f.write(f"Número de iteraciones: {self.model.n_iter_}\n")
            
            if hasattr(self, 'label_encoder') and hasattr(self.label_encoder, 'classes_'):
                f.write(f"\nClases: {list(self.label_encoder.classes_)}\n")
            
            if hasattr(self.X_train, 'shape'):
                f.write(f"\nDimensiones de entrada: {self.X_train.shape[1]}\n")
        
        print(f"  Información del modelo guardada en {os.path.join(self.output_directory, f'{nombre_modelo}_info.txt')}")
    
    def visualizar_arquitectura(self):
        """Genera una visualización de la arquitectura de la red neuronal"""
        if self.model is None:
            print("No hay modelo para visualizar")
            return
        
        try:
            # Obtener información de la arquitectura
            layers = [self.X_train.shape[1]] + list(self.model.hidden_layer_sizes) + [1 if len(np.unique(self.y_train)) <= 2 else len(np.unique(self.y_train))]
            
            # Configuración del gráfico
            plt.figure(figsize=(12, 9))
            
            # Número de neuronas máximo para escala
            max_neurons = max(layers)
            v_spacing = 0.25
            h_spacing = 1
            
            # Dibujar nodos y conexiones
            for n, layer_size in enumerate(layers):
                layer_top = v_spacing * (layer_size - 1) / 2.
                for m in range(layer_size):
                    circle = plt.Circle((n * h_spacing, layer_top - m * v_spacing), v_spacing/4, 
                                       fill=False, color='b' if n < len(layers)-1 else 'r')
                    plt.gca().add_patch(circle)
                    
                    # Añadir texto para capa de entrada y salida
                    if n == 0:
                        plt.text(n * h_spacing - 0.18, layer_top - m * v_spacing, 
                               f"In {m+1}", fontsize=8)
                    elif n == len(layers) - 1:
                        if len(np.unique(self.y_train)) <= 2:
                            plt.text(n * h_spacing + 0.1, layer_top - m * v_spacing, 
                                   "Out", fontsize=8)
                        else:
                            plt.text(n * h_spacing + 0.1, layer_top - m * v_spacing, 
                                   f"Out {m+1}", fontsize=8)
                
                # Dibujar conexiones entre capas
                if n < len(layers) - 1:
                    next_layer_top = v_spacing * (layers[n+1] - 1) / 2.
                    for i in range(layer_size):
                        for j in range(layers[n+1]):
                            alpha = 0.3  # Transparencia
                            plt.plot([n * h_spacing, (n + 1) * h_spacing],
                                   [layer_top - i * v_spacing, next_layer_top - j * v_spacing],
                                   'k-', alpha=alpha)
                
                # Añadir etiquetas de capa
                if n == 0:
                    plt.text(n * h_spacing, -1, "Capa de\nEntrada", fontsize=10, ha='center')
                elif n == len(layers) - 1:
                    plt.text(n * h_spacing, -1, "Capa de\nSalida", fontsize=10, ha='center')
                else:
                    plt.text(n * h_spacing, -1, f"Capa Oculta {n}\n{layer_size} neuronas", fontsize=10, ha='center')
            
            # Ajustar límites
            plt.axis('equal')
            plt.axis('off')
            plt.title(f'Arquitectura del Perceptrón Multicapa\nActivación: {self.model.activation}, Solver: {self.model.solver}')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_directory, 'network_architecture.png'), dpi=300)
            print(f"  Visualización de arquitectura guardada en {os.path.join(self.output_directory, 'network_architecture.png')}")
            
        except Exception as e:
            print(f"Error al visualizar arquitectura: {str(e)}")

# Ejemplo de uso
def main():
    # Configurar rutas
    csv_directory = "./informacion/csv"  # Directorio con archivos CSV
    output_directory = "./models/mlp"    # Directorio para guardar modelos
    
    # Inicializar entrenador
    trainer = MLPModelTrainer(
        csv_directory=csv_directory,
        output_directory=output_directory,
        target_column='Patron_Movimiento',  # Ajustar según el objetivo
        features_to_exclude=['Frame', 'Objeto', 'Video']
    )
    
    # Cargar y preparar datos
    if not trainer.cargar_datos():
        print("Error al cargar datos. Terminando proceso.")
        return
    
    trainer.limpiar_datos()
    
    # Aplicar PCA para reducir dimensionalidad (opcional)
    apply_pca = True
    trainer.preparar_datos_entrenamiento(test_size=0.2, random_state=42, apply_pca=apply_pca)
    
    # Opción 1: Entrenar con hiperparámetros fijos
    trainer.entrenar_modelo(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size='auto',
        learning_rate='adaptive',
        max_iter=500
    )
    
    # Opción 2: Optimizar hiperparámetros (descomenta para usar)
    # best_params = trainer.optimizar_hiperparametros(method='random', n_iter=20)
    
    # Evaluar y guardar el modelo
    trainer.evaluar_modelo()
    trainer.visualizar_arquitectura()
    trainer.guardar_modelo(nombre_modelo='mlp_patrones_movimiento')
    
    print("Proceso de entrenamiento completado.")

if __name__ == "__main__":
    main()