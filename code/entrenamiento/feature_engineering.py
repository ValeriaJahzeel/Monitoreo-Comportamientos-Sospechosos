"""
Módulo para ingeniería de características de comportamientos.
Este script realiza selección y transformación de características para optimizar
los modelos de clasificación.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

class FeatureEngineer:
    def __init__(self, data_dir='./processed_data/'):
        """
        Inicializa el ingeniero de características.
        
        Parámetros:
        -----------
        data_dir : str
            Directorio donde se encuentran los datos procesados
        """
        self.data_dir = data_dir
        self.output_dir = os.path.join(data_dir, 'feature_analysis')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Cargar datos de entrenamiento
        try:
            self.video_train = pd.read_csv(os.path.join(data_dir, 'video_train_scaled.csv'))
            self.object_train = pd.read_csv(os.path.join(data_dir, 'object_train_scaled.csv'))
            self.frame_train = pd.read_csv(os.path.join(data_dir, 'frame_train_scaled.csv'))
        except FileNotFoundError:
            print("Error: No se encontraron los archivos de datos procesados.")
            print("Ejecute primero data_processing.py para generar los archivos necesarios.")
    
    def analyze_correlations(self, level='video'):
        """
        Analiza y visualiza las correlaciones entre características.
        
        Parámetros:
        -----------
        level : str
            Nivel de agregación a analizar ('video', 'object' o 'frame')
        """
        print(f"Analizando correlaciones para características de nivel '{level}'...")
        
        # Seleccionar dataset según nivel
        if level == 'video':
            data = self.video_train
        elif level == 'object':
            data = self.object_train
        elif level == 'frame':
            data = self.frame_train
        else:
            raise ValueError("Nivel no válido. Use 'video', 'object' o 'frame'.")
        
        # Seleccionar solo columnas numéricas
        numeric_data = data.select_dtypes(include=['float64', 'int64'])
        
        # Excluir columnas que no son características
        exclude_cols = ['clase_encoded', 'video_id', 'Frame_', 'Objeto_']
        feature_cols = [col for col in numeric_data.columns if not any(col.startswith(ex) for ex in exclude_cols)]
        
        # Calcular matriz de correlación
        corr_matrix = numeric_data[feature_cols].corr()
        
        # Visualizar matriz de correlación
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm',
                   vmin=-1, vmax=1, center=0, square=True, linewidths=.5)
        plt.title(f'Correlaciones entre Características de {level.capitalize()}')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{level}_correlations.png'))
        plt.close()
        
        # Identificar pares altamente correlacionados (|r| > 0.8)
        strong_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.8:
                    strong_correlations.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': corr_matrix.iloc[i, j]
                    })
        
        # Guardar correlaciones fuertes en un CSV
        if strong_correlations:
            strong_corr_df = pd.DataFrame(strong_correlations)
            strong_corr_df.sort_values('correlation', ascending=False, inplace=True)
            strong_corr_df.to_csv(os.path.join(self.output_dir, f'{level}_strong_correlations.csv'), index=False)
            print(f"Se encontraron {len(strong_correlations)} pares de características altamente correlacionadas")
        else:
            print("No se encontraron características altamente correlacionadas")
        
        return corr_matrix
    
    def select_features_univariate(self, level='video', method='f_classif', k=10):
        """
        Selecciona las mejores características usando métodos univariados.
        
        Parámetros:
        -----------
        level : str
            Nivel de agregación a analizar ('video', 'object' o 'frame')
        method : str
            Método de selección ('f_classif' o 'mutual_info')
        k : int
            Número de características a seleccionar
        """
        print(f"Seleccionando las mejores {k} características para nivel '{level}' usando {method}...")
        
        # Seleccionar dataset según nivel
        if level == 'video':
            data = self.video_train
        elif level == 'object':
            data = self.object_train
        elif level == 'frame':
            data = self.frame_train
        else:
            raise ValueError("Nivel no válido. Use 'video', 'object' o 'frame'.")
        
        # Seleccionar solo columnas numéricas
        numeric_data = data.select_dtypes(include=['float64', 'int64'])
        
        # Excluir columnas que no son características
        exclude_cols = ['clase_encoded', 'video_id', 'Frame_', 'Objeto_']
        feature_cols = [col for col in numeric_data.columns if not any(col.startswith(ex) for ex in exclude_cols)]
        
        # Preparar datos
        X = numeric_data[feature_cols]
        y = data['clase_encoded']
        
        # Elegir método de selección
        if method == 'f_classif':
            selector = SelectKBest(f_classif, k=min(k, len(feature_cols)))
            score_func = f_classif
        elif method == 'mutual_info':
            selector = SelectKBest(mutual_info_classif, k=min(k, len(feature_cols)))
            score_func = mutual_info_classif
        else:
            raise ValueError("Método no válido. Use 'f_classif' o 'mutual_info'.")
        
        # Ajustar selector
        selector.fit(X, y)
        
        # Obtener puntuaciones y p-valores
        scores = selector.scores_
        if method == 'f_classif':
            p_values = selector.pvalues_
        else:
            p_values = [None] * len(scores)  # No hay p-valores para mutual_info
        
        # Crear DataFrame con resultados
        results = pd.DataFrame({
            'feature': feature_cols,
            'score': scores,
            'p_value': p_values
        })
        
        # Ordenar por puntuación
        results.sort_values('score', ascending=False, inplace=True)
        
        # Guardar resultados
        results.to_csv(os.path.join(self.output_dir, f'{level}_{method}_scores.csv'), index=False)
        
        # Visualizar
        plt.figure(figsize=(12, 6))
        sns.barplot(x='score', y='feature', data=results.head(min(k, len(feature_cols))))
        plt.title(f'Top {k} Características para {level.capitalize()} usando {method}')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{level}_{method}_top{k}.png'))
        plt.close()
        
        # Devolver las mejores características
        top_features = results.head(min(k, len(feature_cols)))['feature'].tolist()
        
        return top_features, results
    
    def select_features_rf_importance(self, level='video', n_estimators=100, k=10):
        """
        Selecciona características usando importancia de Random Forest.
        
        Parámetros:
        -----------
        level : str
            Nivel de agregación a analizar ('video', 'object' o 'frame')
        n_estimators : int
            Número de árboles en el bosque
        k : int
            Número de características a seleccionar
        """
        print(f"Seleccionando las mejores {k} características para nivel '{level}' usando Random Forest...")
        
        # Seleccionar dataset según nivel
        if level == 'video':
            data = self.video_train
        elif level == 'object':
            data = self.object_train
        elif level == 'frame':
            data = self.frame_train
        else:
            raise ValueError("Nivel no válido. Use 'video', 'object' o 'frame'.")
        
        # Seleccionar solo columnas numéricas
        numeric_data = data.select_dtypes(include=['float64', 'int64'])
        
        # Excluir columnas que no son características
        exclude_cols = ['clase_encoded', 'video_id', 'Frame_', 'Objeto_']
        feature_cols = [col for col in numeric_data.columns if not any(col.startswith(ex) for ex in exclude_cols)]
        
        # Preparar datos
        X = numeric_data[feature_cols]
        y = data['clase_encoded']
        
        # Entrenar Random Forest
        rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        rf.fit(X, y)
        
        # Calcular importancia de características
        importances = rf.feature_importances_
        std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
        
        # Crear DataFrame con resultados
        results = pd.DataFrame({
            'feature': feature_cols,
            'importance': importances,
            'std': std
        })
        
        # Ordenar por importancia
        results.sort_values('importance', ascending=False, inplace=True)
        
        # Guardar resultados
        results.to_csv(os.path.join(self.output_dir, f'{level}_rf_importance.csv'), index=False)
        
        # Visualizar
        plt.figure(figsize=(12, 6))
        sns.barplot(x='importance', y='feature', data=results.head(min(k, len(feature_cols))))
        plt.title(f'Top {k} Características para {level.capitalize()} usando Random Forest')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{level}_rf_importance_top{k}.png'))
        plt.close()
        
        # Calcular también importancia por permutación
        perm_importance = permutation_importance(rf, X, y, n_repeats=10, random_state=42)
        
        # Crear DataFrame con resultados de importancia por permutación
        perm_results = pd.DataFrame({
            'feature': feature_cols,
            'importance': perm_importance.importances_mean,
            'std': perm_importance.importances_std
        })
        
        # Ordenar por importancia
        perm_results.sort_values('importance', ascending=False, inplace=True)
        
        # Guardar resultados
        perm_results.to_csv(os.path.join(self.output_dir, f'{level}_perm_importance.csv'), index=False)
        
        # Visualizar
        plt.figure(figsize=(12, 6))
        sns.barplot(x='importance', y='feature', data=perm_results.head(min(k, len(feature_cols))))
        plt.title(f'Top {k} Características para {level.capitalize()} usando Importancia por Permutación')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{level}_perm_importance_top{k}.png'))
        plt.close()
        
        # Devolver las mejores características de ambos métodos
        top_features_rf = results.head(min(k, len(feature_cols)))['feature'].tolist()
        top_features_perm = perm_results.head(min(k, len(feature_cols)))['feature'].tolist()
        
        # Unir características únicas de ambos métodos
        top_features = list(set(top_features_rf + top_features_perm))
        
        return top_features, (results, perm_results)
    
    def apply_pca(self, level='video', n_components=0.95):
        """
        Aplica PCA para reducción de dimensionalidad.
        
        Parámetros:
        -----------
        level : str
            Nivel de agregación a analizar ('video', 'object' o 'frame')
        n_components : float o int
            Si es float, indica la varianza acumulada deseada (0-1)
            Si es int, indica el número de componentes a retener
        """
        print(f"Aplicando PCA para nivel '{level}'...")
        
        # Seleccionar dataset según nivel
        if level == 'video':
            data = self.video_train
        elif level == 'object':
            data = self.object_train
        elif level == 'frame':
            data = self.frame_train
        else:
            raise ValueError("Nivel no válido. Use 'video', 'object' o 'frame'.")
        
        # Seleccionar solo columnas numéricas
        numeric_data = data.select_dtypes(include=['float64', 'int64'])
        
        # Excluir columnas que no son características
        exclude_cols = ['clase_encoded', 'video_id', 'Frame_', 'Objeto_']
        feature_cols = [col for col in numeric_data.columns if not any(col.startswith(ex) for ex in exclude_cols)]
        
        # Preparar datos
        X = numeric_data[feature_cols]
        
        # Aplicar PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        
        # Calcular varianza acumulada
        var_ratio = pca.explained_variance_ratio_
        cumulative_var = np.cumsum(var_ratio)
        
        # Crear DataFrame con componentes principales
        pca_cols = [f'PC{i+1}' for i in range(X_pca.shape[1])]
        pca_df = pd.DataFrame(X_pca, columns=pca_cols)
        
        # Añadir información de clase y video
        pca_df['clase_encoded'] = data['clase_encoded']
        pca_df['video_id'] = data['video_id']
        if 'clase_first' in data.columns:
            pca_df['clase'] = data['clase_first']
        
        # Guardar resultados
        pca_df.to_csv(os.path.join(self.output_dir, f'{level}_pca.csv'), index=False)
        
        # Guardar información de componentes
        component_df = pd.DataFrame(pca.components_, columns=feature_cols, index=pca_cols)
        component_df.to_csv(os.path.join(self.output_dir, f'{level}_pca_components.csv'))
        
        # Visualizar varianza explicada
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, len(var_ratio) + 1), var_ratio, alpha=0.6, label='Varianza individual')
        plt.plot(range(1, len(cumulative_var) + 1), cumulative_var, 'r-', marker='o', label='Varianza acumulada')
        plt.axhline(y=0.95, color='g', linestyle='--', label='Umbral 95%')
        plt.xlabel('Componentes Principales')
        plt.ylabel('Proporción de Varianza Explicada')
        plt.title(f'Varianza Explicada por Componentes Principales - {level.capitalize()}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{level}_pca_variance.png'))
        plt.close()
        
        # Visualizar proyección 2D
        if X_pca.shape[1] >= 2:
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=data['clase_encoded'], cmap='viridis', alpha=0.7)
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            plt.title(f'Proyección PCA 2D - {level.capitalize()}')
            plt.colorbar(scatter, label='Clase')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'{level}_pca_projection.png'))
            plt.close()
        
        return pca, pca_df, component_df
    
    def get_optimal_feature_set(self, level='video', method='combined', n_features=15):
        """
        Obtiene un conjunto óptimo de características, combinando múltiples métodos.
        
        Parámetros:
        -----------
        level : str
            Nivel de agregación a analizar ('video', 'object' o 'frame')
        method : str
            Método de selección ('f_classif', 'mutual_info', 'rf', 'combined')
        n_features : int
            Número máximo de características a seleccionar
        """
        print(f"Obteniendo conjunto óptimo de características para nivel '{level}'...")
        
        if method == 'f_classif':
            features, _ = self.select_features_univariate(level, method='f_classif', k=n_features)
        elif method == 'mutual_info':
            features, _ = self.select_features_univariate(level, method='mutual_info', k=n_features)
        elif method == 'rf':
            features, _ = self.select_features_rf_importance(level, k=n_features)
        elif method == 'combined':
            # Ejecutar todos los métodos
            features_f, _ = self.select_features_univariate(level, method='f_classif', k=n_features)
            features_mi, _ = self.select_features_univariate(level, method='mutual_info', k=n_features)
            features_rf, _ = self.select_features_rf_importance(level, k=n_features)
            
            # Contar ocurrencias de cada característica
            all_features = features_f + features_mi + features_rf
            feature_counts = pd.Series(all_features).value_counts()
            
            # Seleccionar características que aparecen en al menos 2 métodos
            features_common = feature_counts[feature_counts >= 2].index.tolist()
            
            # Si no hay suficientes características comunes, agregar las más importantes de cada método
            if len(features_common) < n_features:
                remaining = n_features - len(features_common)
                
                # Características que solo aparecen en un método, ordenadas por la suma de sus rankings
                features_single = feature_counts[feature_counts == 1].index.tolist()
                
                # Añadir características únicas hasta completar
                features = features_common + features_single[:remaining]
            else:
                features = features_common[:n_features]
            
            # Guardar resultado de la combinación
            feature_info = pd.DataFrame({
                'feature': feature_counts.index,
                'count': feature_counts.values
            })
            feature_info.sort_values('count', ascending=False, inplace=True)
            feature_info.to_csv(os.path.join(self.output_dir, f'{level}_combined_features.csv'), index=False)
        else:
            raise ValueError("Método no válido. Use 'f_classif', 'mutual_info', 'rf' o 'combined'.")
        
        print(f"Conjunto óptimo seleccionado con {len(features)} características")
        return features
    
    def analyze_all_levels(self):
        """
        Realiza análisis de características para todos los niveles de agregación.
        """
        results = {}
        
        for level in ['video', 'object', 'frame']:
            print(f"\n{'='*50}")
            print(f"ANÁLISIS DE CARACTERÍSTICAS PARA NIVEL '{level.upper()}'")
            print(f"{'='*50}")
            
            # Analizar correlaciones
            corr_matrix = self.analyze_correlations(level)
            
            # Aplicar métodos de selección de características
            f_features, f_scores = self.select_features_univariate(level, method='f_classif')
            mi_features, mi_scores = self.select_features_univariate(level, method='mutual_info')
            rf_features, rf_scores = self.select_features_rf_importance(level)
            
            # Aplicar PCA
            pca, pca_df, pca_components = self.apply_pca(level)
            
            # Obtener conjunto óptimo
            optimal_features = self.get_optimal_feature_set(level, method='combined')
            
            # Almacenar resultados
            results[level] = {
                'correlations': corr_matrix,
                'f_classif': (f_features, f_scores),
                'mutual_info': (mi_features, mi_scores),
                'random_forest': (rf_features, rf_scores),
                'pca': (pca, pca_df, pca_components),
                'optimal_features': optimal_features
            }
        
        return results

if __name__ == "__main__":
    engineer = FeatureEngineer()
    engineer.analyze_all_levels()