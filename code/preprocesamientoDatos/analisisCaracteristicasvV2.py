"""
Script para optimizar características y entrenar un modelo con el conjunto óptimo
Basado en los resultados del análisis exploratorio
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import classification_report, confusion_matrix
import glob
import joblib
import warnings
warnings.filterwarnings('ignore')

class FeatureOptimizer:
    def __init__(self, csv_directory, output_directory='./modelo_optimizado'):
        """
        Inicializa el optimizador de características
        
        Parámetros:
        -----------
        csv_directory : str
            Ruta al directorio que contiene los archivos CSV para entrenamiento
        output_directory : str
            Directorio donde se guardarán el modelo y resultados
        """
        self.csv_directory = csv_directory
        self.output_directory = output_directory
        self.dataframes = {}  # Almacenar DataFrames por archivo
        self.combined_data = None  # Para análisis global
        self.selected_features = None  # Características seleccionadas
        self.best_model = None  # Mejor modelo entrenado
        
        # Crear directorio de salida si no existe
        os.makedirs(output_directory, exist_ok=True)
    
    def cargar_datos(self):
        """Carga todos los archivos CSV del directorio especificado"""
        csv_files = glob.glob(os.path.join(self.csv_directory, "*.csv"))
        
        if not csv_files:
            print(f"No se encontraron archivos CSV en {self.csv_directory}")
            return False
        
        print(f"Cargando {len(csv_files)} archivos CSV...")
        
        for csv_file in csv_files:
            nombre_video = os.path.basename(csv_file).split('.')[0]
            try:
                df = pd.read_csv(csv_file)
                # Añadir columna de identificación del video
                df['Video'] = nombre_video
                self.dataframes[nombre_video] = df
                print(f"  ✓ Cargado {nombre_video}: {len(df)} registros")
            except Exception as e:
                print(f"  ✗ Error al cargar {nombre_video}: {str(e)}")
        
        # Crear DataFrame combinado para análisis global
        if self.dataframes:
            self.combined_data = pd.concat(self.dataframes.values(), ignore_index=True)
            print(f"Total de registros combinados: {len(self.combined_data)}")
            return True
        return False
    
    def limpiar_datos(self):
        """Limpia y preprocesa los datos para análisis"""
        if self.combined_data is None:
            print("No hay datos cargados para limpiar")
            return
        
        print("Limpiando y preprocesando datos...")
        
        # Reemplazar valores infinitos o NaN
        self.combined_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Imputar valores faltantes con la mediana para columnas numéricas
        columnas_numericas = self.combined_data.select_dtypes(include=['float64', 'int64']).columns
        for col in columnas_numericas:
            self.combined_data[col].fillna(self.combined_data[col].median(), inplace=True)
        
        # Imputar valores categóricos con el valor más frecuente
        columnas_categoricas = self.combined_data.select_dtypes(include=['object']).columns
        for col in columnas_categoricas:
            if col != 'Video':  # No imputar columna de video
                self.combined_data[col].fillna(self.combined_data[col].mode()[0], inplace=True)
    
    def obtener_caracteristicas_target(self, target_column='Patron_Movimiento', excludir=None):
        """
        Obtiene las características y la variable objetivo
        
        Parámetros:
        -----------
        target_column : str
            Nombre de la columna objetivo
        excludir : list
            Lista de columnas a excluir del análisis
        """
        if self.combined_data is None:
            print("No hay datos cargados")
            return None, None, None
        
        # Lista predeterminada de columnas a excluir
        if excludir is None:
            excludir = ['Frame', 'Objeto', 'Video']
        
        # Filtrar columnas que no sean la objetivo y no estén en la lista de exclusión
        caracteristicas = [col for col in self.combined_data.columns 
                          if col != target_column and col not in excludir]
        
        # Obtener X e y
        X = self.combined_data[caracteristicas]
        y = self.combined_data[target_column]
        
        return X, y, caracteristicas
    
    def seleccionar_caracteristicas(self, target_column='Patron_Movimiento', metodo='mutual_info', 
                                  n_caracteristicas=None, umbral_score=0.01):
        """
        Selecciona las características más relevantes usando diferentes métodos
        
        Parámetros:
        -----------
        target_column : str
            Nombre de la columna objetivo
        metodo : str
            Método de selección ('mutual_info', 'random_forest')
        n_caracteristicas : int
            Número de características a seleccionar (si es None, se determina automáticamente)
        umbral_score : float
            Umbral mínimo para considerar una característica relevante
        """
        if self.combined_data is None:
            print("No hay datos cargados")
            return None
        
        print(f"\nSeleccionando características óptimas usando método: {metodo}")
        
        # Obtener características y target
        X, y, caracteristicas = self.obtener_caracteristicas_target(target_column)
        
        # Seleccionar solo características numéricas
        X_num = X.select_dtypes(include=['float64', 'int64'])
        
        # Diferentes métodos de selección
        if metodo == 'mutual_info':
            # Calcular información mutua
            mi_scores = mutual_info_classif(X_num, y, random_state=42)
            
            # Crear DataFrame con scores
            scores_df = pd.DataFrame({
                'Caracteristica': X_num.columns,
                'Score': mi_scores
            }).sort_values('Score', ascending=False)
            
        elif metodo == 'random_forest':
            # Usar Random Forest para importancia de características
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_num, y)
            
            # Crear DataFrame con scores
            scores_df = pd.DataFrame({
                'Caracteristica': X_num.columns,
                'Score': rf.feature_importances_
            }).sort_values('Score', ascending=False)
            
        else:
            print(f"Método no reconocido: {metodo}")
            return None
        
        # Determinar número de características si no se especificó
        if n_caracteristicas is None:
            # Seleccionar características con score superior al umbral
            n_caracteristicas = sum(scores_df['Score'] > umbral_score)
            
            # Asegurar un mínimo de características
            n_caracteristicas = max(n_caracteristicas, 5)
            
            print(f"Número de características determinado automáticamente: {n_caracteristicas}")
        
        # Seleccionar las mejores características
        selected_features = scores_df.head(n_caracteristicas)['Caracteristica'].tolist()
        self.selected_features = selected_features
        
        # Mostrar características seleccionadas
        print(f"Seleccionadas {len(selected_features)} características:")
        for i, feature in enumerate(selected_features):
            score = scores_df[scores_df['Caracteristica'] == feature]['Score'].values[0]
            print(f"  {i+1}. {feature}: {score:.4f}")
        
        # Guardar lista de características
        pd.DataFrame({'Caracteristica': selected_features}).to_csv(
            os.path.join(self.output_directory, 'caracteristicas_seleccionadas.csv'),
            index=False
        )
        
        return selected_features
    
    def evaluar_seleccion(self, target_column='Patron_Movimiento', cross_validation=True):
        """
        Evalúa la calidad de la selección de características
        
        Parámetros:
        -----------
        target_column : str
            Nombre de la columna objetivo
        cross_validation : bool
            Si es True, usa validación cruzada para evaluar
        """
        if self.selected_features is None:
            print("Primero debe seleccionar características usando seleccionar_caracteristicas()")
            return
        
        if self.combined_data is None:
            print("No hay datos cargados")
            return
        
        print("\nEvaluando selección de características...")
        
        # Obtener características y target
        X, y, _ = self.obtener_caracteristicas_target(target_column)
        
        # Seleccionar solo características numéricas
        X_num = X.select_dtypes(include=['float64', 'int64'])
        
        # Dividir datos (para comparación con/sin selección)
        X_train, X_test, y_train, y_test = train_test_split(
            X_num, y, test_size=0.3, random_state=42, stratify=y if len(y.unique()) > 1 else None
        )
        
        # Preparar modelos para evaluación
        rf_all = RandomForestClassifier(n_estimators=100, random_state=42)
        mlp_all = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            alpha=0.001,
            max_iter=500,
            random_state=42
        )
        
        # Seleccionar subconjunto de características
        X_train_selected = X_train[self.selected_features]
        X_test_selected = X_test[self.selected_features]
        
        # Preparar modelos para características seleccionadas
        rf_selected = RandomForestClassifier(n_estimators=100, random_state=42)
        mlp_selected = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            alpha=0.001,
            max_iter=500,
            random_state=42
        )
        
        # Evaluar con validación cruzada o train/test
        results = {}
        
        if cross_validation:
            # Evaluación con validación cruzada
            print("Evaluando con validación cruzada (5-fold)...")
            
            # Random Forest - Todas las características
            rf_all_scores = cross_val_score(rf_all, X_num, y, cv=5)
            results['RF-todas'] = rf_all_scores.mean()
            
            # MLP - Todas las características
            mlp_all_scores = cross_val_score(mlp_all, X_num, y, cv=5)
            results['MLP-todas'] = mlp_all_scores.mean()
            
            # Random Forest - Características seleccionadas
            rf_selected_scores = cross_val_score(
                rf_selected, X_num[self.selected_features], y, cv=5
            )
            results['RF-seleccionadas'] = rf_selected_scores.mean()
            
            # MLP - Características seleccionadas
            mlp_selected_scores = cross_val_score(
                mlp_selected, X_num[self.selected_features], y, cv=5
            )
            results['MLP-seleccionadas'] = mlp_selected_scores.mean()
            
        else:
            # Evaluación con train/test split
            print("Evaluando con train/test split (70/30)...")
            
            # Random Forest - Todas las características
            rf_all.fit(X_train, y_train)
            results['RF-todas'] = rf_all.score(X_test, y_test)
            
            # MLP - Todas las características
            mlp_all.fit(X_train, y_train)
            results['MLP-todas'] = mlp_all.score(X_test, y_test)
            
            # Random Forest - Características seleccionadas
            rf_selected.fit(X_train_selected, y_train)
            results['RF-seleccionadas'] = rf_selected.score(X_test_selected, y_test)
            
            # MLP - Características seleccionadas
            mlp_selected.fit(X_train_selected, y_train)
            results['MLP-seleccionadas'] = mlp_selected.score(X_test_selected, y_test)
        
        # Mostrar resultados
        print("\nResultados de evaluación:")
        print(f"  Random Forest (todas las características): {results['RF-todas']:.4f}")
        print(f"  Random Forest (características seleccionadas): {results['RF-seleccionadas']:.4f}")
        print(f"  MLP (todas las características): {results['MLP-todas']:.4f}")
        print(f"  MLP (características seleccionadas): {results['MLP-seleccionadas']:.4f}")
        
        # Determinar mejor modelo
        mejor_score = max(results.values())
        mejor_modelo = [k for k, v in results.items() if v == mejor_score][0]
        
        print(f"\nMejor modelo: {mejor_modelo} (precisión: {mejor_score:.4f})")
        
        # Entrenar el mejor modelo con todos los datos
        print("\nEntrenando modelo final con todos los datos...")
        
        if mejor_modelo == 'RF-todas':
            final_model = RandomForestClassifier(n_estimators=100, random_state=42)
            final_model.fit(X_num, y)
            self.selected_features = list(X_num.columns)
        elif mejor_modelo == 'RF-seleccionadas':
            final_model = RandomForestClassifier(n_estimators=100, random_state=42)
            final_model.fit(X_num[self.selected_features], y)
        elif mejor_modelo == 'MLP-todas':
            final_model = MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.001,
                max_iter=500,
                random_state=42
            )
            final_model.fit(X_num, y)
            self.selected_features = list(X_num.columns)
        else:  # MLP-seleccionadas
            final_model = MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.001,
                max_iter=500,
                random_state=42
            )
            final_model.fit(X_num[self.selected_features], y)
        
        self.best_model = final_model
        
        # Guardar modelo y características
        self.guardar_modelo(mejor_modelo.split('-')[0])
        
        return results, mejor_modelo
    
    def guardar_modelo(self, tipo_modelo):
        """
        Guarda el modelo entrenado y las características seleccionadas
        
        Parámetros:
        -----------
        tipo_modelo : str
            Tipo de modelo (RF, MLP)
        """
        if self.best_model is None or self.selected_features is None:
            print("No hay modelo o características para guardar")
            return
        
        print("\nGuardando modelo y características seleccionadas...")
        
        # Guardar modelo
        model_path = os.path.join(self.output_directory, f'modelo_optimizado_{tipo_modelo}.pkl')
        joblib.dump(self.best_model, model_path)
        
        # Guardar lista de características
        features_path = os.path.join(self.output_directory, 'caracteristicas_seleccionadas.txt')
        with open(features_path, 'w') as f:
            for feature in self.selected_features:
                f.write(f"{feature}\n")
        
        # Guardar información del modelo
        info_path = os.path.join(self.output_directory, 'info_modelo.txt')
        with open(info_path, 'w') as f:
            f.write(f"Tipo de modelo: {tipo_modelo}\n")
            f.write(f"Número de características: {len(self.selected_features)}\n")
            f.write(f"Características seleccionadas: {', '.join(self.selected_features)}\n")
            
            # Añadir parámetros específicos del modelo
            f.write("\nParámetros del modelo:\n")
            for param, value in self.best_model.get_params().items():
                f.write(f"  {param}: {value}\n")
        
        print(f"  ✓ Modelo guardado en {model_path}")
        print(f"  ✓ Características guardadas en {features_path}")
        print(f"  ✓ Información del modelo guardada en {info_path}")
    
    def optimizar_caracteristicas_entrenar_modelo(self, target_column='Patron_Movimiento'):
        """
        Proceso completo de optimización de características y entrenamiento de modelo
        
        Parámetros:
        -----------
        target_column : str
            Nombre de la columna objetivo
        """
        # 1. Cargar y limpiar datos
        if not hasattr(self, 'combined_data') or self.combined_data is None:
            if not self.cargar_datos():
                print("No se pudieron cargar los datos.")
                return
            self.limpiar_datos()
        
        # 2. Seleccionar características usando información mutua
        self.seleccionar_caracteristicas(
            target_column=target_column,
            metodo='mutual_info',
            n_caracteristicas=None  # Determinar automáticamente
        )
        
        # 3. Evaluar la selección y entrenar modelo final
        results, mejor_modelo = self.evaluar_seleccion(
            target_column=target_column,
            cross_validation=True
        )
        
        print("\nProceso de optimización y entrenamiento completado.")
        print(f"Modelo final: {mejor_modelo}")
        print(f"Características seleccionadas: {len(self.selected_features)}")
        
        return self.best_model, self.selected_features

# Función principal de ejemplo
def optimizar_y_entrenar(directorio_csv, directorio_salida, target_column='Patron_Movimiento'):
    """
    Función principal para optimizar características y entrenar un modelo
    
    Parámetros:
    -----------
    directorio_csv : str
        Directorio con los archivos CSV para entrenamiento
    directorio_salida : str
        Directorio donde se guardarán el modelo y resultados
    target_column : str
        Columna objetivo para la clasificación
    """
    print("=" * 60)
    print("OPTIMIZACIÓN DE CARACTERÍSTICAS Y ENTRENAMIENTO DE MODELO")
    print("=" * 60)
    
    # Inicializar optimizador
    optimizer = FeatureOptimizer(
        csv_directory=directorio_csv,
        output_directory=directorio_salida
    )
    
    # Ejecutar proceso completo
    mejor_modelo, caracteristicas_seleccionadas = optimizer.optimizar_caracteristicas_entrenar_modelo(
        target_column=target_column
    )
    
    print("\n" + "=" * 60)
    print(f"PROCESO COMPLETADO. Resultados guardados en {directorio_salida}")
    print("=" * 60)
    
    return optimizer, mejor_modelo, caracteristicas_seleccionadas

def derivar_nuevas_caracteristicas(dataframe, features_base=None):
    """
    Deriva nuevas características a partir de las existentes
    
    Parámetros:
    -----------
    dataframe : DataFrame
        DataFrame con los datos originales
    features_base : list
        Lista de características base a utilizar (si es None, usa todas las numéricas)
    
    Retorna:
    --------
    DataFrame con características originales y derivadas
    """
    # Crear copia para no modificar el original
    df = dataframe.copy()
    
    # Si no se especifican características base, usar todas las numéricas excepto Frame y Objeto
    if features_base is None:
        features_base = [col for col in df.select_dtypes(include=['float64', 'int64']).columns
                        if col not in ['Frame', 'Objeto']]
    
    print(f"Derivando nuevas características a partir de {len(features_base)} características base...")
    
    # 1. Características de ratio entre pares relevantes
    pares_ratio = [
        ('Velocidad', 'Aceleracion'),
        ('Linealidad', 'Circularidad'),
        ('Zigzag', 'Linealidad'),
        ('Velocidad', 'Desplazamiento'),
        ('Area_Trayectoria', 'Desplazamiento')
    ]
    
    for col1, col2 in pares_ratio:
        if col1 in df.columns and col2 in df.columns:
            nombre_ratio = f"Ratio_{col1}_{col2}"
            # Evitar divisiones por cero
            denominador = df[col2].replace(0, np.nan)
            df[nombre_ratio] = df[col1] / denominador
            # Reemplazar infinitos con NaN y luego imputar
            df[nombre_ratio].replace([np.inf, -np.inf], np.nan, inplace=True)
            df[nombre_ratio].fillna(df[nombre_ratio].median(), inplace=True)
            print(f"  ✓ Creada característica: {nombre_ratio}")
    
    # 2. Productos entre características relevantes
    pares_producto = [
        ('Velocidad', 'Zigzag'),
        ('Circularidad', 'Area_Trayectoria'),
        ('Aceleracion', 'Zigzag'),
        ('Es_Ciclico', 'Frecuencia_Ciclo')
    ]
    
    for col1, col2 in pares_producto:
        if col1 in df.columns and col2 in df.columns:
            nombre_producto = f"Producto_{col1}_{col2}"
            df[nombre_producto] = df[col1] * df[col2]
            print(f"  ✓ Creada característica: {nombre_producto}")
    
    # 3. Características cuadráticas para capturar relaciones no lineales
    for col in ['Velocidad', 'Aceleracion', 'Linealidad', 'Circularidad', 'Zigzag']:
        if col in df.columns:
            nombre_cuadratico = f"{col}_Cuadrado"
            df[nombre_cuadratico] = df[col] ** 2
            print(f"  ✓ Creada característica: {nombre_cuadratico}")
    
    # 4. Normalización por objeto (variación relativa)
    for col in ['Velocidad', 'Aceleracion', 'Desplazamiento']:
        if col in df.columns:
            nombre_normalizado = f"{col}_Normalizado"
            # Normalizar por objeto
            df[nombre_normalizado] = df.groupby('Objeto')[col].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-5)
            )
            print(f"  ✓ Creada característica: {nombre_normalizado}")
    
    # 5. Características compuestas específicas para comportamientos
    if all(col in df.columns for col in ['Zigzag', 'Velocidad', 'Aceleracion']):
        # Índice de erraticidad (combinación de zigzag y aceleración)
        df['Indice_Erraticidad'] = df['Zigzag'] * df['Aceleracion'].abs() / (df['Velocidad'] + 1e-5)
        print(f"  ✓ Creada característica: Indice_Erraticidad")
    
    if all(col in df.columns for col in ['Circularidad', 'Es_Ciclico', 'Frecuencia_Ciclo']):
        # Índice de circularidad cíclica
        df['Indice_Circularidad_Ciclica'] = df['Circularidad'] * df['Es_Ciclico'] * (df['Frecuencia_Ciclo'] + 0.1)
        print(f"  ✓ Creada característica: Indice_Circularidad_Ciclica")
    
    if all(col in df.columns for col in ['Linealidad', 'Velocidad', 'Desplazamiento']):
        # Índice de propósito (movimientos lineales y constantes)
        df['Indice_Proposito'] = df['Linealidad'] * df['Velocidad'] / (df['Velocidad'].std() + 1e-5)
        print(f"  ✓ Creada característica: Indice_Proposito")
    
    # 6. Características temporales (si hay suficientes frames)
    if 'Frame' in df.columns:
        # Duración de la trayectoria por objeto
        df['Duracion_Trayectoria'] = df.groupby('Objeto')['Frame'].transform(
            lambda x: x.max() - x.min() + 1
        )
        print(f"  ✓ Creada característica: Duracion_Trayectoria")
        
        # Densidad temporal (cantidad de frames por unidad de desplazamiento)
        if 'Desplazamiento' in df.columns:
            desplazamiento_total = df.groupby('Objeto')['Desplazamiento'].transform('sum')
            df['Densidad_Temporal'] = df['Duracion_Trayectoria'] / (desplazamiento_total + 1e-5)
            print(f"  ✓ Creada característica: Densidad_Temporal")
    
    print(f"\nSe derivaron {len(df.columns) - len(dataframe.columns)} nuevas características.")
    return df

def ejecutar_analisis_completo(directorio_csv, directorio_analisis, directorio_modelo, 
                            target_column='Patron_Movimiento', derivar_caracteristicas=True):
    """
    Ejecuta el proceso completo de análisis, optimización y entrenamiento
    
    Parámetros:
    -----------
    directorio_csv : str
        Directorio con los archivos CSV para análisis y entrenamiento
    directorio_analisis : str
        Directorio donde se guardarán los resultados del análisis exploratorio
    directorio_modelo : str
        Directorio donde se guardará el modelo optimizado
    target_column : str
        Columna objetivo para la clasificación
    derivar_caracteristicas : bool
        Si es True, deriva nuevas características antes del análisis
    """
    from exploratory_analysis import FeatureExplorer
    
    print("\n" + "=" * 60)
    print("INICIANDO ANÁLISIS COMPLETO DE CARACTERÍSTICAS")
    print("=" * 60)
    
    # Paso 1: Derivar nuevas características (opcional)
    if derivar_caracteristicas:
        print("Paso 1: Derivando nuevas características...")
        
        # Cargar datos originales
        temp_optimizer = FeatureOptimizer(directorio_csv)
        temp_optimizer.cargar_datos()
        
        # Derivar nuevas características
        df_derivado = derivar_nuevas_caracteristicas(temp_optimizer.combined_data)
        
        # Guardar en directorio temporal
        directorio_temp = os.path.join(directorio_csv, "derivado")
        os.makedirs(directorio_temp, exist_ok=True)
        
        # Guardar por video
        for video, video_df in temp_optimizer.dataframes.items():
            # Derivar características para este video
            video_derivado = derivar_nuevas_caracteristicas(video_df)
            # Guardar
            video_derivado.to_csv(os.path.join(directorio_temp, f"{video}_derivado.csv"), index=False)
        
        # Usar directorio temporal para análisis
        directorio_csv_analisis = directorio_temp
    else:
        directorio_csv_analisis = directorio_csv
    
    # Paso 2: Realizar análisis exploratorio
    print("\nPaso 2: Realizando análisis exploratorio...")
    explorer = FeatureExplorer(
        csv_directory=directorio_csv_analisis,
        output_directory=directorio_analisis
    )
    explorer.generar_informe_completo(target_column)
    
    # Paso 3: Optimizar características y entrenar modelo
    print("\nPaso 3: Optimizando características y entrenando modelo...")
    optimizer, model, selected_features = optimizar_y_entrenar(
        directorio_csv=directorio_csv_analisis,
        directorio_salida=directorio_modelo,
        target_column=target_column
    )
    
    print("\n" + "=" * 60)
    print("PROCESO COMPLETO FINALIZADO")
    print("=" * 60)
    print(f"Análisis exploratorio: {directorio_analisis}")
    print(f"Modelo optimizado: {directorio_modelo}")
    print(f"Características seleccionadas: {len(selected_features)}")
    
    return explorer, optimizer, model

# Si se ejecuta como script principal
if __name__ == "__main__":
    import argparse
    
    # Configurar argumentos
    parser = argparse.ArgumentParser(
        description='Optimización de características y entrenamiento de modelo'
    )
    parser.add_argument('--csv-dir', required=True, help='Directorio con archivos CSV')
    parser.add_argument('--output-dir', default='./modelo_optimizado', 
                     help='Directorio para resultados')
    parser.add_argument('--target', default='Patron_Movimiento', 
                     help='Columna objetivo para la clasificación')
    parser.add_argument('--derivar', action='store_true', 
                     help='Derivar nuevas características')
    
    args = parser.parse_args()
    
    # Ejecutar optimización
    optimizar_y_entrenar(
        directorio_csv=args.csv_dir,
        directorio_salida=args.output_dir,
        target_column=args.target
    )