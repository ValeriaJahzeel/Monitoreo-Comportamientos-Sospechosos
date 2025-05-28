"""
Analizador de características mejorado que maneja datos originales y agregados.
Selecciona características relevantes para cada tipo de dataset.
Versión corregida con mejor manejo de errores y estructura modular.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import warnings
import json
from pathlib import Path
warnings.filterwarnings('ignore')

# Configurar estilo de matplotlib
plt.style.use('default')
sns.set_palette("husl")

class EnhancedFeatureAnalyzer:
    def __init__(self, data_dir='./processed_data/', output_dir='./feature_analysis/', 
                 use_original_data=True, use_aggregated_data=True):
        """
        Inicializa el analizador mejorado.
        
        Parámetros:
        -----------
        data_dir : str
            Directorio con datos procesados
        output_dir : str
            Directorio para guardar análisis
        use_original_data : bool
            Si analizar datos originales
        use_aggregated_data : bool
            Si analizar datos agregados
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.use_original_data = use_original_data
        self.use_aggregated_data = use_aggregated_data
        
        # Crear directorios
        self._create_directories()
        
        # Cargar metadatos
        self.metadata = self._load_metadata()
        self.classes = self.metadata.get('classes', ['normal', 'forcejeo', 'merodeo'])
        
        print(f"🔍 Analizador mejorado inicializado")
        print(f"   Datos: {self.data_dir}")
        print(f"   Salida: {self.output_dir}")
        print(f"   Datos originales: {'✅' if use_original_data else '❌'}")
        print(f"   Datos agregados: {'✅' if use_aggregated_data else '❌'}")
    
    def _create_directories(self):
        """Crea directorios necesarios."""
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            (self.output_dir / 'plots').mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"⚠️ Error creando directorios: {e}")
            raise
    
    def _load_metadata(self):
        """Carga metadatos del procesamiento."""
        try:
            # Intentar cargar JSON primero
            json_path = self.data_dir / 'metadata.json'
            if json_path.exists():
                with open(json_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            
            # Fallback a CSV
            csv_path = self.data_dir / 'metadata.csv'
            if csv_path.exists():
                return pd.read_csv(csv_path).iloc[0].to_dict()
            
            print("⚠️ No se encontraron metadatos")
            return {}
        except Exception as e:
            print(f"⚠️ Error cargando metadatos: {e}")
            return {}
    
    def get_available_datasets(self):
        """Obtiene lista de datasets disponibles."""
        available_datasets = []
        
        # Verificar datos originales
        if self.use_original_data:
            original_train = self.data_dir / 'original_train_scaled.csv'
            if original_train.exists():
                available_datasets.append('original')
        
        # Verificar datos agregados
        if self.use_aggregated_data:
            for level in ['video', 'object', 'frame']:
                level_train = self.data_dir / f'{level}_train_scaled.csv'
                if level_train.exists():
                    available_datasets.append(level)
        
        print(f"📊 Datasets disponibles para análisis: {available_datasets}")
        return available_datasets
    
    def load_training_data(self, dataset_name='original'):
        """
        Carga SOLO datos de entrenamiento para análisis.
        
        Parámetros:
        -----------
        dataset_name : str
            Nombre del dataset ('original', 'video', 'object', 'frame')
        """
        try:
            train_path = self.data_dir / f'{dataset_name}_train_scaled.csv'
            
            if not train_path.exists():
                print(f"❌ No se encontró: {train_path}")
                return None, None, None, None
            
            train_data = pd.read_csv(train_path)
            
            if train_data.empty:
                print(f"❌ Archivo vacío: {train_path}")
                return None, None, None, None
            
            print(f"⚠️ USANDO SOLO DATOS DE ENTRENAMIENTO para {dataset_name}")
            print(f"   Esto evita data leakage con el conjunto de test")
            
            # Identificar variable objetivo
            target_col = self._identify_target_column(train_data, dataset_name)
            if target_col is None:
                return None, None, None, None
            
            # Identificar características
            feature_cols = self._identify_feature_columns(train_data, target_col)
            
            if len(feature_cols) == 0:
                print(f"❌ No se encontraron características numéricas en {dataset_name}")
                return None, None, None, None
            
            # Extraer X y y
            X = train_data[feature_cols].copy()
            y = train_data[target_col].copy()
            
            # Limpiar datos
            X, y = self._clean_feature_data(X, y)
            
            print(f"📊 Datos cargados para '{dataset_name}':")
            print(f"   Muestras: {len(X)}")
            print(f"   Características: {len(feature_cols)}")
            print(f"   Clases únicas: {len(y.unique())}")
            print(f"   Distribución de clases: {dict(y.value_counts())}")
            
            return X, y, feature_cols, train_data
            
        except Exception as e:
            print(f"❌ Error cargando {dataset_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None, None, None
    
    def _identify_target_column(self, data, dataset_name):
        """Identifica la columna objetivo."""
        # Buscar columnas de clase
        target_candidates = [
            'clase_encoded', 
            'clase_encoded_first',
            'clase_first'
        ]
        
        for col in target_candidates:
            if col in data.columns:
                return col
        
        # Buscar cualquier columna que contenga 'clase'
        clase_cols = [col for col in data.columns if 'clase' in col.lower()]
        
        if clase_cols:
            target_col = clase_cols[0]
            print(f"  📌 Usando '{target_col}' como variable objetivo")
            
            # Si es categórica, codificar
            if data[target_col].dtype == 'object':
                le = LabelEncoder()
                try:
                    data['clase_encoded'] = le.fit_transform(data[target_col])
                    print(f"  🔄 Codificado '{target_col}' -> 'clase_encoded'")
                    return 'clase_encoded'
                except Exception as e:
                    print(f"  ❌ Error codificando: {e}")
                    return None
            else:
                return target_col
        
        print(f"❌ No se encontró columna de clase en {dataset_name}")
        print(f"   Columnas disponibles: {list(data.columns)}")
        return None
    
    def _identify_feature_columns(self, data, target_col):
        """Identifica columnas de características."""
        exclude_patterns = [
            'video_id', 'Objeto', 'Frame', 'clase', 'split', 
            'index', 'unnamed', target_col
        ]
        
        # Obtener columnas numéricas
        numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
        
        # Filtrar columnas excluidas
        feature_cols = []
        for col in numeric_cols:
            if not any(pattern.lower() in col.lower() for pattern in exclude_patterns):
                feature_cols.append(col)
        
        print(f"  📊 Características identificadas: {len(feature_cols)}")
        if len(feature_cols) > 20:
            print(f"     Ejemplos: {feature_cols[:10]}...")
        else:
            print(f"     Todas: {feature_cols}")
        
        return feature_cols
    
    def _clean_feature_data(self, X, y):
        """Limpia datos de características."""
        # Verificar y limpiar X
        original_shape = X.shape
        
        # Reemplazar infinitos
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # Llenar NaN
        X = X.fillna(0)
        
        # Eliminar columnas constantes
        constant_cols = X.columns[X.nunique() <= 1]
        if len(constant_cols) > 0:
            print(f"  🗑️ Eliminando {len(constant_cols)} columnas constantes")
            X = X.drop(columns=constant_cols)
        
        # Verificar que y es válido
        y = y.dropna()
        
        # Asegurar que X e y tienen el mismo índice
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]
        
        print(f"  🧹 Datos limpiados: {original_shape} -> {X.shape}")
        
        return X, y
    
    def analyze_feature_importance_rf(self, X, y, feature_names, dataset_name='dataset', n_top=15):
        """Analiza importancia usando Random Forest."""
        print(f"🌲 Analizando importancia RF para {dataset_name}...")
        
        try:
            # Entrenar Random Forest
            rf = RandomForestClassifier(
                n_estimators=100, 
                random_state=42, 
                n_jobs=-1,
                max_depth=10  # Limitar profundidad para evitar overfitting
            )
            rf.fit(X, y)
            
            # Importancias básicas
            importances = rf.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            rf_importance = pd.DataFrame({
                'feature': [feature_names[i] for i in indices],
                'importance': importances[indices],
                'rank': range(1, len(feature_names) + 1)
            })
            
            # Importancia por permutación (con manejo de errores)
            try:
                print("  🔄 Calculando importancia por permutación...")
                perm_importance = permutation_importance(
                    rf, X, y, n_repeats=3, random_state=42, n_jobs=-1
                )
                perm_df = pd.DataFrame({
                    'feature': feature_names,
                    'perm_importance': perm_importance.importances_mean,
                    'perm_std': perm_importance.importances_std
                }).sort_values('perm_importance', ascending=False)
                
                # Combinar resultados
                combined_rf = rf_importance.merge(perm_df, on='feature')
                
            except Exception as e:
                print(f"  ⚠️ Error en permutation importance: {e}")
                combined_rf = rf_importance.copy()
                combined_rf['perm_importance'] = combined_rf['importance']
                combined_rf['perm_std'] = 0
            
            # Guardar resultados
            output_file = self.output_dir / f'{dataset_name}_rf_importance.csv'
            combined_rf.to_csv(output_file, index=False)
            
            # Visualizar
            self._plot_rf_importance(combined_rf, dataset_name, n_top)
            
            print(f"  ✅ Análisis RF completado: {len(combined_rf)} características")
            
            return combined_rf
            
        except Exception as e:
            print(f"  ❌ Error en análisis RF: {e}")
            # Retornar DataFrame vacío en caso de error
            return pd.DataFrame({
                'feature': feature_names,
                'importance': 0,
                'rank': range(1, len(feature_names) + 1),
                'perm_importance': 0,
                'perm_std': 0
            })
    
    def _plot_rf_importance(self, combined_rf, dataset_name, n_top):
        """Crea visualización de importancia RF."""
        try:
            plt.figure(figsize=(12, 8))
            
            # Subplot 1: Importancia RF
            plt.subplot(2, 1, 1)
            top_features = combined_rf.head(n_top)
            colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
            bars = plt.barh(range(len(top_features)), top_features['importance'], color=colors)
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Importancia RF')
            plt.title(f'Top {n_top} Características - Random Forest ({dataset_name})')
            plt.gca().invert_yaxis()
            
            # Subplot 2: Importancia por permutación
            plt.subplot(2, 1, 2)
            if 'perm_importance' in top_features.columns:
                bars = plt.barh(range(len(top_features)), top_features['perm_importance'], 
                               color=colors, alpha=0.7)
                plt.yticks(range(len(top_features)), top_features['feature'])
                plt.xlabel('Importancia por Permutación')
                plt.title(f'Top {n_top} Características - Permutation Importance ({dataset_name})')
                plt.gca().invert_yaxis()
            
            plt.tight_layout()
            plot_path = self.output_dir / 'plots' / f'{dataset_name}_rf_importance.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"    ⚠️ Error creando gráfico RF: {e}")
            plt.close()
    
    def analyze_statistical_significance(self, X, y, feature_names, dataset_name='dataset', alpha=0.05):
        """Analiza significancia estadística."""
        print(f"📈 Analizando significancia estadística para {dataset_name}...")
        
        try:
            # F-test
            f_scores, f_pvalues = f_classif(X, y)
            
            # Mutual Information
            mi_scores = mutual_info_classif(X, y, random_state=42)
            
            # DataFrame con resultados
            stats_df = pd.DataFrame({
                'feature': feature_names,
                'f_score': f_scores,
                'f_pvalue': f_pvalues,
                'mi_score': mi_scores,
                'is_significant': f_pvalues < alpha
            })
            
            stats_df = stats_df.sort_values('f_score', ascending=False)
            
            # Guardar resultados
            output_file = self.output_dir / f'{dataset_name}_statistical_analysis.csv'
            stats_df.to_csv(output_file, index=False)
            
            # Visualizar
            self._plot_statistical_analysis(stats_df, dataset_name)
            
            print(f"  ✅ Análisis estadístico completado")
            print(f"     Características significativas: {stats_df['is_significant'].sum()}")
            
            return stats_df
            
        except Exception as e:
            print(f"  ❌ Error en análisis estadístico: {e}")
            # Retornar DataFrame con valores por defecto
            return pd.DataFrame({
                'feature': feature_names,
                'f_score': 0,
                'f_pvalue': 1,
                'mi_score': 0,
                'is_significant': False
            })
    
    def _plot_statistical_analysis(self, stats_df, dataset_name):
        """Crea visualización de análisis estadístico."""
        try:
            plt.figure(figsize=(14, 6))
            
            # F-scores
            plt.subplot(1, 2, 1)
            top_f = stats_df.head(15)
            colors = ['red' if not sig else 'green' for sig in top_f['is_significant']]
            bars = plt.barh(range(len(top_f)), top_f['f_score'], color=colors, alpha=0.7)
            plt.yticks(range(len(top_f)), top_f['feature'])
            plt.xlabel('F-Score')
            plt.title(f'F-Scores Top 15 ({dataset_name})')
            plt.gca().invert_yaxis()
            
            # Mutual Information
            plt.subplot(1, 2, 2)
            top_mi = stats_df.nlargest(15, 'mi_score')
            bars = plt.barh(range(len(top_mi)), top_mi['mi_score'], 
                           color=plt.cm.coolwarm(np.linspace(0, 1, len(top_mi))))
            plt.yticks(range(len(top_mi)), top_mi['feature'])
            plt.xlabel('Mutual Information')
            plt.title(f'Mutual Information Top 15 ({dataset_name})')
            plt.gca().invert_yaxis()
            
            plt.tight_layout()
            plot_path = self.output_dir / 'plots' / f'{dataset_name}_statistical.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"    ⚠️ Error creando gráfico estadístico: {e}")
            plt.close()
    
    def recursive_feature_elimination(self, X, y, feature_names, dataset_name='dataset', n_features=10):
        """Selección recursiva de características."""
        print(f"🔄 RFE para {dataset_name}...")
        
        try:
            # Ajustar n_features según el dataset
            max_features = min(len(feature_names), 20)
            n_features = min(n_features, max_features)
            
            estimator = LogisticRegression(max_iter=1000, random_state=42)
            rfe = RFE(estimator, n_features_to_select=n_features, step=1)
            
            rfe.fit(X, y)
            
            rfe_df = pd.DataFrame({
                'feature': feature_names,
                'selected': rfe.support_,
                'ranking': rfe.ranking_
            }).sort_values('ranking')
            
            # Guardar resultados
            output_file = self.output_dir / f'{dataset_name}_rfe_results.csv'
            rfe_df.to_csv(output_file, index=False)
            
            # Visualizar
            self._plot_rfe_results(rfe_df, dataset_name)
            
            print(f"  ✅ RFE completado: {rfe_df['selected'].sum()} características seleccionadas")
            
            return rfe_df
            
        except Exception as e:
            print(f"  ❌ Error en RFE: {e}")
            return pd.DataFrame({
                'feature': feature_names, 
                'selected': False, 
                'ranking': 1
            })
    
    def _plot_rfe_results(self, rfe_df, dataset_name):
        """Crea visualización de resultados RFE."""
        try:
            selected_features = rfe_df[rfe_df['selected']].head(15)
            
            if len(selected_features) > 0:
                plt.figure(figsize=(10, 6))
                colors = plt.cm.Set2(np.linspace(0, 1, len(selected_features)))
                bars = plt.barh(range(len(selected_features)), selected_features['ranking'], 
                               color=colors)
                plt.yticks(range(len(selected_features)), selected_features['feature'])
                plt.xlabel('Ranking RFE')
                plt.title(f'Características Seleccionadas - RFE ({dataset_name})')
                plt.gca().invert_yaxis()
                plt.tight_layout()
                
                plot_path = self.output_dir / 'plots' / f'{dataset_name}_rfe.png'
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                
        except Exception as e:
            print(f"    ⚠️ Error creando gráfico RFE: {e}")
            plt.close()
    
    def correlation_analysis(self, X, feature_names, dataset_name='dataset', threshold=0.8):
        """Analiza correlaciones entre características."""
        print(f"🔗 Analizando correlaciones para {dataset_name}...")
        
        try:
            # Calcular matriz de correlación
            corr_matrix = X.corr()
            
            # Encontrar correlaciones altas
            high_corr_pairs = []
            n_features = len(corr_matrix.columns)
            
            for i in range(n_features):
                for j in range(i+1, n_features):
                    corr_val = corr_matrix.iloc[i, j]
                    if not np.isnan(corr_val) and abs(corr_val) > threshold:
                        high_corr_pairs.append({
                            'feature1': corr_matrix.columns[i],
                            'feature2': corr_matrix.columns[j],
                            'correlation': corr_val,
                            'abs_correlation': abs(corr_val)
                        })
            
            high_corr_df = pd.DataFrame(high_corr_pairs)
            if len(high_corr_df) > 0:
                high_corr_df = high_corr_df.sort_values('abs_correlation', ascending=False)
            
            # Guardar resultados
            output_file = self.output_dir / f'{dataset_name}_high_correlations.csv'
            high_corr_df.to_csv(output_file, index=False)
            
            # Visualizar
            self._plot_correlation_matrix(corr_matrix, feature_names, dataset_name)
            
            print(f"  ✅ Análisis de correlación completado")
            print(f"     Correlaciones altas encontradas: {len(high_corr_df)}")
            
            return corr_matrix, high_corr_df
            
        except Exception as e:
            print(f"  ❌ Error en análisis de correlación: {e}")
            # Retornar matrices vacías
            empty_corr = pd.DataFrame(np.eye(len(feature_names)), 
                                    index=feature_names, columns=feature_names)
            return empty_corr, pd.DataFrame()
    
    def _plot_correlation_matrix(self, corr_matrix, feature_names, dataset_name):
        """Crea visualización de matriz de correlación."""
        try:
            plt.figure(figsize=(12, 10))
            
            # Seleccionar subset para visualización
            if len(feature_names) > 20:
                # Usar características con mayor varianza
                variances = corr_matrix.var().sort_values(ascending=False)
                top_features = variances.head(20).index
                corr_subset = corr_matrix.loc[top_features, top_features]
            else:
                corr_subset = corr_matrix
            
            # Crear máscara triangular
            mask = np.triu(np.ones_like(corr_subset, dtype=bool))
            
            # Heatmap
            sns.heatmap(corr_subset, mask=mask, annot=True, fmt='.2f', 
                       cmap='coolwarm', vmin=-1, vmax=1, center=0,
                       square=True, cbar_kws={"shrink": .8})
            plt.title(f'Matriz de Correlación ({dataset_name})')
            plt.tight_layout()
            
            plot_path = self.output_dir / 'plots' / f'{dataset_name}_correlation.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"    ⚠️ Error creando matriz de correlación: {e}")
            plt.close()
    
    def select_optimal_features(self, dataset_name='original', max_features=15):
        """Selecciona características óptimas combinando métodos."""
        print(f"⭐ Seleccionando características óptimas para {dataset_name}...")
        
        # Cargar datos
        X, y, feature_names, _ = self.load_training_data(dataset_name)
        if X is None:
            print(f"❌ No se pudieron cargar datos para {dataset_name}")
            return None
        
        # Ajustar max_features según dataset
        max_features = min(max_features, len(feature_names), 25)
        
        # Ejecutar análisis
        try:
            rf_results = self.analyze_feature_importance_rf(X, y, feature_names, dataset_name)
            stats_results = self.analyze_statistical_significance(X, y, feature_names, dataset_name)
            rfe_results = self.recursive_feature_elimination(X, y, feature_names, dataset_name, max_features)
            corr_matrix, high_corr = self.correlation_analysis(X, feature_names, dataset_name)
            
            # Combinar resultados
            optimal_features = self._combine_feature_selection_results(
                rf_results, stats_results, rfe_results, high_corr, 
                feature_names, dataset_name, max_features
            )
            
            return optimal_features
            
        except Exception as e:
            print(f"❌ Error en selección de características: {e}")
            return None
    
    def _combine_feature_selection_results(self, rf_results, stats_results, rfe_results, 
                                         high_corr, feature_names, dataset_name, max_features):
        """Combina resultados de diferentes métodos de selección."""
        try:
            # Crear DataFrame base
            feature_scores = pd.DataFrame({'feature': feature_names})
            
            # RF scores (normalizado)
            rf_scores = rf_results[['feature', 'importance']].copy()
            rf_scores['rf_score'] = rf_scores['importance'] / (rf_scores['importance'].max() + 1e-8)
            feature_scores = feature_scores.merge(rf_scores[['feature', 'rf_score']], on='feature', how='left')
            feature_scores['rf_score'] = feature_scores['rf_score'].fillna(0)
            
            # Statistical scores
            stats_scores = stats_results[['feature', 'f_score', 'mi_score', 'is_significant']].copy()
            f_max = stats_scores['f_score'].max() + 1e-8
            mi_max = stats_scores['mi_score'].max() + 1e-8
            stats_scores['stat_score'] = (
                (stats_scores['f_score'] / f_max) * 0.7 +
                (stats_scores['mi_score'] / mi_max) * 0.3
            )
            feature_scores = feature_scores.merge(
                stats_scores[['feature', 'stat_score', 'is_significant']], on='feature', how='left'
            )
            feature_scores['stat_score'] = feature_scores['stat_score'].fillna(0)
            feature_scores['is_significant'] = feature_scores['is_significant'].fillna(False)
            
            # RFE scores
            rfe_scores = rfe_results[['feature', 'ranking', 'selected']].copy()
            max_rank = rfe_scores['ranking'].max()
            rfe_scores['rfe_score'] = (max_rank - rfe_scores['ranking'] + 1) / max_rank
            feature_scores = feature_scores.merge(
                rfe_scores[['feature', 'rfe_score', 'selected']], on='feature', how='left'
            )
            feature_scores['rfe_score'] = feature_scores['rfe_score'].fillna(0)
            feature_scores['selected'] = feature_scores['selected'].fillna(False)
            
            # Penalizar correlaciones altas
            feature_scores['corr_penalty'] = 0
            if len(high_corr) > 0:
                for _, row in high_corr.iterrows():
                    if row['abs_correlation'] > 0.8:
                        mask1 = feature_scores['feature'] == row['feature1']
                        mask2 = feature_scores['feature'] == row['feature2']
                        feature_scores.loc[mask1, 'corr_penalty'] += 0.1
                        feature_scores.loc[mask2, 'corr_penalty'] += 0.1
            
            # Score final
            feature_scores['final_score'] = (
                feature_scores['rf_score'] * 0.4 +
                feature_scores['stat_score'] * 0.3 +
                feature_scores['rfe_score'] * 0.2 +
                feature_scores['is_significant'].astype(int) * 0.1 -
                feature_scores['corr_penalty']
            )
            
            # Ordenar y seleccionar
            feature_scores = feature_scores.sort_values('final_score', ascending=False)
            selected_features = feature_scores.head(max_features)
            
            # Guardar resultados
            feature_scores_file = self.output_dir / f'{dataset_name}_feature_selection.csv'
            feature_scores.to_csv(feature_scores_file, index=False)
            
            selected_features_file = self.output_dir / f'{dataset_name}_selected_features.csv'
            selected_features.to_csv(selected_features_file, index=False)
            
            # Visualizar selección final
            self._plot_final_selection(selected_features, dataset_name)
            
            print(f"  ✅ {len(selected_features)} características seleccionadas para {dataset_name}")
            
            return selected_features['feature'].tolist()
            
        except Exception as e:
            print(f"❌ Error combinando resultados: {e}")
            return feature_names[:max_features]  # Fallback
    
    def _plot_final_selection(self, selected_features, dataset_name):
        """Crea visualización de la selección final."""
        try:
            plt.figure(figsize=(12, 8))
            colors = plt.cm.viridis(np.linspace(0, 1, len(selected_features)))
            bars = plt.barh(range(len(selected_features)), selected_features['final_score'], 
                           color=colors)
            plt.yticks(range(len(selected_features)), selected_features['feature'])
            plt.xlabel('Score Final')
            plt.title(f'Características Seleccionadas - Score Final ({dataset_name})')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            
            plot_path = self.output_dir / 'plots' / f'{dataset_name}_final_selection.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"    ⚠️ Error creando gráfico de selección final: {e}")
            plt.close()
    
    def generate_dataset_report(self, dataset_name='original'):
        """Genera reporte completo para un dataset."""
        print(f"📄 Generando reporte para {dataset_name}...")
        
        try:
            # Seleccionar características
            optimal_features = self.select_optimal_features(dataset_name)
            
            if optimal_features is None:
                print(f"❌ No se pudieron seleccionar características para {dataset_name}")
                return None
            
            # Cargar resultados
            feature_scores_file = self.output_dir / f'{dataset_name}_feature_selection.csv'
            if not feature_scores_file.exists():
                print(f"❌ No se encontró archivo de scores: {feature_scores_file}")
                return optimal_features
            
            feature_scores = pd.read_csv(feature_scores_file)
            
            # Crear reporte HTML
            html_content = self._create_html_report(dataset_name, feature_scores, optimal_features)
            
            # Guardar reporte
            report_path = self.output_dir / f'{dataset_name}_feature_report.html'
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            print(f"  ✅ Reporte guardado: {report_path}")
            
            return optimal_features
            
        except Exception as e:
            print(f"❌ Error generando reporte: {e}")
            return optimal_features if 'optimal_features' in locals() else None
    
    def _create_html_report(self, dataset_name, feature_scores, optimal_features):
        """Crea contenido HTML del reporte."""
        dataset_description = {
            'original': 'Datos originales sin agregación - máxima granularidad',
            'video': 'Datos agregados por video - patrones de comportamiento completo',
            'object': 'Datos agregados por objeto - comportamiento de objetos específicos',
            'frame': 'Datos agregados por frame - evolución temporal'
        }
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Análisis de Características - {dataset_name}</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                       margin: 20px; background-color: #f8f9fa; line-height: 1.6; }}
                .container {{ max-width: 1200px; margin: 0 auto; background-color: white; 
                           padding: 30px; border-radius: 15px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); }}
                h1 {{ color: #2c3e50; text-align: center; border-bottom: 3px solid #3498db; 
                     padding-bottom: 15px; margin-bottom: 30px; }}
                h2 {{ color: #34495e; border-left: 4px solid #3498db; padding-left: 15px; 
                     margin-top: 30px; }}
                h3 {{ color: #2c3e50; margin-top: 25px; }}
                .summary {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                           color: white; padding: 25px; border-radius: 15px; margin: 25px 0; }}
                .metric {{ display: inline-block; margin: 15px 25px; text-align: center; }}
                .metric-value {{ font-size: 2.2em; font-weight: bold; display: block; }}
                .metric-label {{ font-size: 0.9em; opacity: 0.9; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; 
                        box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background: linear-gradient(135deg, #3498db, #2980b9); color: white; 
                     font-weight: 600; }}
                tr:nth-child(even) {{ background-color: #f8f9fa; }}
                tr:hover {{ background-color: #e8f4fd; transition: background-color 0.3s; }}
                .selected {{ background-color: #d4edda !important; font-weight: 500; }}
                .selected:hover {{ background-color: #c3e6cb !important; }}
                img {{ max-width: 100%; height: auto; margin: 20px 0; border-radius: 10px; 
                      box-shadow: 0 4px 15px rgba(0,0,0,0.1); }}
                .dataset-type {{ background: linear-gradient(135deg, #e8f4fd, #f0f8ff); 
                               padding: 20px; border-radius: 10px; margin: 20px 0; 
                               border-left: 5px solid #3498db; }}
                .feature-card {{ background-color: #ffffff; border: 1px solid #e1e8ed; 
                               border-radius: 8px; padding: 15px; margin: 10px 0; 
                               box-shadow: 0 2px 8px rgba(0,0,0,0.05); }}
                .score-bar {{ background: linear-gradient(90deg, #3498db, #2ecc71); 
                            height: 20px; border-radius: 10px; margin: 5px 0; }}
                .footer {{ text-align: center; color: #95a5a6; margin-top: 50px; 
                          padding-top: 20px; border-top: 1px solid #ecf0f1; }}
                .alert {{ background-color: #fff3cd; border: 1px solid #ffeaa7; 
                         color: #856404; padding: 15px; border-radius: 8px; margin: 15px 0; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🔍 Análisis de Características</h1>
                <div class="dataset-type">
                    <h3>📊 Dataset: {dataset_name.upper()}</h3>
                    <p>{dataset_description.get(dataset_name, 'Dataset personalizado')}</p>
                </div>
                
                <div class="summary">
                    <h2 style="color: white; border: none; padding: 0;">📊 Resumen del Análisis</h2>
                    <div class="metric">
                        <div class="metric-value">{len(feature_scores)}</div>
                        <div class="metric-label">Características Analizadas</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{len(optimal_features)}</div>
                        <div class="metric-label">Características Seleccionadas</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">4</div>
                        <div class="metric-label">Métodos de Selección</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{feature_scores['is_significant'].sum()}</div>
                        <div class="metric-label">Estadísticamente Significativas</div>
                    </div>
                </div>
                
                <div class="alert">
                    <strong>⚠️ Importante:</strong> Este análisis se basa únicamente en datos de entrenamiento 
                    para evitar data leakage. Los datos de test permanecen completamente reservados.
                </div>
                
                <h2>🏆 Características Seleccionadas</h2>
                <table>
                    <tr>
                        <th>Ranking</th>
                        <th>Característica</th>
                        <th>Score Final</th>
                        <th>RF Score</th>
                        <th>Stat Score</th>
                        <th>RFE Score</th>
                        <th>Significativa</th>
                        <th>Penalización Corr.</th>
                    </tr>
        """
        
        # Añadir características seleccionadas
        top_features = feature_scores.head(len(optimal_features))
        for i, (_, row) in enumerate(top_features.iterrows()):
            html_content += f"""
                <tr class="selected">
                    <td><strong>{i+1}</strong></td>
                    <td><strong>{row['feature']}</strong></td>
                    <td>{row['final_score']:.3f}</td>
                    <td>{row.get('rf_score', 0):.3f}</td>
                    <td>{row.get('stat_score', 0):.3f}</td>
                    <td>{row.get('rfe_score', 0):.3f}</td>
                    <td>{'✅' if row.get('is_significant', False) else '❌'}</td>
                    <td>{row.get('corr_penalty', 0):.3f}</td>
                </tr>
            """
        
        html_content += """
                </table>
                
                <h2>📊 Análisis Detallado</h2>
                <h3>🔬 Metodología de Selección</h3>
                <div class="feature-card">
                    <strong>1. Random Forest Importance (40%)</strong>
                    <p>Mide la importancia de cada característica en la predicción usando bosques aleatorios.</p>
                </div>
                <div class="feature-card">
                    <strong>2. Análisis Estadístico (30%)</strong>
                    <p>Combina F-scores y Mutual Information para evaluar relevancia estadística.</p>
                </div>
                <div class="feature-card">
                    <strong>3. Recursive Feature Elimination (20%)</strong>
                    <p>Selección iterativa eliminando características menos importantes.</p>
                </div>
                <div class="feature-card">
                    <strong>4. Significancia Estadística (10%)</strong>
                    <p>Bonus para características estadísticamente significativas (p < 0.05).</p>
                </div>
                <div class="feature-card">
                    <strong>5. Penalización por Correlación</strong>
                    <p>Reduce el score de características altamente correlacionadas para evitar redundancia.</p>
                </div>
                
                <h2>📈 Visualizaciones</h2>
        """
        
        # Añadir imágenes si existen
        plots = [
            (f'{dataset_name}_rf_importance.png', '🌲 Importancia Random Forest'),
            (f'{dataset_name}_statistical.png', '📊 Análisis Estadístico'),
            (f'{dataset_name}_rfe.png', '🔄 Selección RFE'),
            (f'{dataset_name}_correlation.png', '🔗 Matriz de Correlación'),
            (f'{dataset_name}_final_selection.png', '⭐ Selección Final')
        ]
        
        for plot_file, title in plots:
            plot_path = self.output_dir / 'plots' / plot_file
            if plot_path.exists():
                html_content += f"""
                <h3>{title}</h3>
                <img src="plots/{plot_file}" alt="{title}">
                """
        
        # Estadísticas adicionales
        html_content += f"""
                <h2>📋 Estadísticas Adicionales</h2>
                <div class="feature-card">
                    <h4>🎯 Distribución de Scores</h4>
                    <p><strong>Score máximo:</strong> {feature_scores['final_score'].max():.3f}</p>
                    <p><strong>Score mínimo:</strong> {feature_scores['final_score'].min():.3f}</p>
                    <p><strong>Score promedio:</strong> {feature_scores['final_score'].mean():.3f}</p>
                    <p><strong>Desviación estándar:</strong> {feature_scores['final_score'].std():.3f}</p>
                </div>
        """
        
        # Correlaciones altas si existen
        corr_file = self.output_dir / f'{dataset_name}_high_correlations.csv'
        if corr_file.exists():
            try:
                high_corr = pd.read_csv(corr_file)
                if len(high_corr) > 0:
                    html_content += """
                    <div class="feature-card">
                        <h4>⚠️ Correlaciones Altas Detectadas</h4>
                        <p>Se encontraron las siguientes correlaciones altas entre características:</p>
                        <ul>
                    """
                    for _, row in high_corr.head(5).iterrows():
                        html_content += f"""
                        <li><strong>{row['feature1']}</strong> ↔ <strong>{row['feature2']}</strong>: 
                            {row['correlation']:.3f}</li>
                        """
                    html_content += "</ul></div>"
            except:
                pass
        
        html_content += """
                <div class="footer">
                    <p>📅 Generado automáticamente por el Sistema Mejorado de Análisis de Características</p>
                    <p>🔬 Análisis basado únicamente en datos de entrenamiento para prevenir data leakage</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html_content
    
    def analyze_all_available_datasets(self):
        """Analiza todas las características disponibles."""
        print("🔍 INICIANDO ANÁLISIS COMPLETO DE CARACTERÍSTICAS")
        print("="*60)
        
        try:
            available_datasets = self.get_available_datasets()
            
            if not available_datasets:
                print("❌ No se encontraron datasets disponibles")
                return {}
            
            results = {}
            
            for dataset_name in available_datasets:
                print(f"\n📊 Analizando dataset: {dataset_name.upper()}")
                print("-" * 40)
                
                try:
                    optimal_features = self.generate_dataset_report(dataset_name)
                    results[dataset_name] = optimal_features
                    
                    if optimal_features:
                        print(f"✅ {dataset_name}: {len(optimal_features)} características seleccionadas")
                    else:
                        print(f"❌ {dataset_name}: Análisis falló")
                        
                except Exception as e:
                    print(f"❌ Error en {dataset_name}: {str(e)}")
                    results[dataset_name] = None
            
            # Generar resumen global
            self._generate_global_summary(results)
            
            print(f"\n🎉 ANÁLISIS COMPLETADO")
            successful_analyses = len([r for r in results.values() if r is not None])
            print(f"   Datasets analizados exitosamente: {successful_analyses}/{len(available_datasets)}")
            print(f"   Resultados guardados en: {self.output_dir}")
            
            return results
            
        except Exception as e:
            print(f"❌ Error en análisis completo: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def _generate_global_summary(self, results):
        """Genera resumen global de todos los análisis."""
        try:
            summary_data = []
            
            for dataset_name, features in results.items():
                if features:
                    for i, feature in enumerate(features):
                        summary_data.append({
                            'dataset': dataset_name,
                            'feature': feature,
                            'rank': i + 1,
                            'dataset_type': 'original' if dataset_name == 'original' else 'aggregated'
                        })
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                summary_file = self.output_dir / 'feature_selection_global_summary.csv'
                summary_df.to_csv(summary_file, index=False)
                
                # Análisis de características comunes
                feature_counts = summary_df['feature'].value_counts()
                common_features = feature_counts[feature_counts > 1]
                
                print(f"\n📊 RESUMEN GLOBAL:")
                print(f"   Total de características únicas seleccionadas: {summary_df['feature'].nunique()}")
                print(f"   Promedio por dataset: {len(summary_data) / len(results):.1f}")
                
                if len(common_features) > 0:
                    print(f"\n🔗 Características comunes entre datasets:")
                    for feature, count in common_features.head(10).items():
                        datasets = summary_df[summary_df['feature'] == feature]['dataset'].tolist()
                        print(f"   {feature}: aparece en {datasets}")
                
                # Crear resumen HTML global
                self._create_global_html_summary(results, summary_df)
                
        except Exception as e:
            print(f"⚠️ Error generando resumen global: {e}")
    
    def _create_global_html_summary(self, results, summary_df):
        """Crea resumen HTML global."""
        try:
            html_content = """
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>Resumen Global - Análisis de Características</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
                    .container { max-width: 1200px; margin: 0 auto; background-color: white; 
                               padding: 30px; border-radius: 10px; box-shadow: 0 0 20px rgba(0,0,0,0.1); }
                    h1 { color: #2c3e50; text-align: center; }
                    .dataset-summary { margin: 20px 0; padding: 15px; background-color: #f8f9fa; 
                                     border-radius: 8px; }
                    table { width: 100%; border-collapse: collapse; margin: 20px 0; }
                    th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
                    th { background-color: #3498db; color: white; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>🌍 Resumen Global del Análisis de Características</h1>
            """
            
            # Resumen por dataset
            html_content += "<h2>📊 Resumen por Dataset</h2>"
            for dataset_name, features in results.items():
                if features:
                    html_content += f"""
                    <div class="dataset-summary">
                        <h3>{dataset_name.upper()}</h3>
                        <p><strong>Características seleccionadas:</strong> {len(features)}</p>
                        <p><strong>Top 5:</strong> {', '.join(features[:5])}</p>
                        <p><a href="{dataset_name}_feature_report.html">Ver reporte detallado →</a></p>
                    </div>
                    """
            
            # Características más comunes
            feature_counts = summary_df['feature'].value_counts()
            common_features = feature_counts[feature_counts > 1].head(10)
            
            if len(common_features) > 0:
                html_content += """
                <h2>🔗 Características Más Relevantes</h2>
                <table>
                    <tr><th>Característica</th><th>Aparece en # Datasets</th><th>Datasets</th></tr>
                """
                
                for feature, count in common_features.items():
                    datasets = summary_df[summary_df['feature'] == feature]['dataset'].tolist()
                    html_content += f"""
                    <tr>
                        <td><strong>{feature}</strong></td>
                        <td>{count}</td>
                        <td>{', '.join(datasets)}</td>
                    </tr>
                    """
                
                html_content += "</table>"
            
            html_content += """
                    <div style="text-align: center; margin-top: 40px; color: #666;">
                        <p>Generado automáticamente por el Sistema de Análisis de Características</p>
                    </div>
                </div>
            </body>
            </html>
            """
            
            # Guardar resumen global
            global_report = self.output_dir / 'global_feature_analysis_summary.html'
            with open(global_report, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            print(f"  ✅ Resumen global guardado: {global_report}")
            
        except Exception as e:
            print(f"⚠️ Error creando resumen HTML global: {e}")

def main():
    """Función principal para ejecutar el analizador."""
    
    # Configuración del analizador
    analyzer = EnhancedFeatureAnalyzer(
        data_dir='./processed_data_enhanced/',
        output_dir='./feature_analysis_enhanced/',
        use_original_data=True,
        use_aggregated_data=True
    )
    
    # Ejecutar análisis completo
    results = analyzer.analyze_all_available_datasets()
    
    if results:
        print(f"\n🎯 ANÁLISIS EXITOSO!")
        
        successful_results = {k: v for k, v in results.items() if v is not None}
        
        if successful_results:
            print(f"   Datasets analizados: {list(successful_results.keys())}")
            
            # Mostrar resumen de características seleccionadas
            for dataset_name, features in successful_results.items():
                print(f"   {dataset_name}: {len(features)} características")
                if len(features) > 0:
                    print(f"     Top 3: {features[:3]}")
        else:
            print("   ⚠️ Ningún dataset se analizó exitosamente")
    
    else:
        print(f"\n❌ ANÁLISIS FALLÓ!")
        
    return results

if __name__ == "__main__":
    results = main()