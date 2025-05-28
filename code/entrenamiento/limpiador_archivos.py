"""
Limpiador completo de datos para análisis de comportamientos.
Maneja valores infinitos, NaN, y convierte características de texto a numéricas.
Versión corregida con mejor manejo de errores y robustez mejorada.
"""
import pandas as pd
import numpy as np
import os
import glob
import json
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import warnings
from typing import Optional, Dict, List, Tuple, Any
warnings.filterwarnings('ignore')

class ComprehensiveDataCleaner:
    """
    Limpiador integral que maneja todos los problemas de datos comunes.
    """
    
    def __init__(self, verbose: bool = True):
        """
        Inicializa el limpiador con configuraciones por defecto.
        
        Parámetros:
        -----------
        verbose : bool
            Si mostrar información detallada durante el procesamiento
        """
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()
        self.numeric_imputer = SimpleImputer(strategy='median')
        self.categorical_imputer = SimpleImputer(strategy='most_frequent')
        self.verbose = verbose
        
        # Estadísticas de limpieza
        self.cleaning_stats = {
            'files_processed': 0,
            'files_successful': 0,
            'files_failed': 0,
            'total_rows_before': 0,
            'total_rows_after': 0,
            'nan_values_filled': 0,
            'inf_values_replaced': 0,
            'text_columns_converted': 0
        }
        
        if self.verbose:
            print("🧹 Limpiador Completo de Datos inicializado")
    
    def clean_single_csv(self, csv_path: str, class_name: Optional[str] = None, 
                        video_id: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Limpia un archivo CSV individual.
        
        Parámetros:
        -----------
        csv_path : str
            Ruta al archivo CSV
        class_name : str, opcional
            Nombre de la clase (normal, forcejeo, sospechoso)
        video_id : str, opcional
            ID del video (se extrae del nombre si no se proporciona)
        """
        csv_path = Path(csv_path)
        
        if not csv_path.exists():
            if self.verbose:
                print(f"❌ Archivo no encontrado: {csv_path}")
            return None
        
        if self.verbose:
            print(f"🧹 Limpiando: {csv_path.name}")
        
        try:
            # Cargar CSV con manejo de errores
            df = self._load_csv_safely(csv_path)
            if df is None:
                return None
            
            original_shape = df.shape
            self.cleaning_stats['total_rows_before'] += original_shape[0]
            
            # Agregar metadatos
            df = self._add_metadata(df, csv_path, class_name, video_id)
            
            if self.verbose:
                print(f"  📊 Datos originales: {original_shape}")
            
            # Pipeline de limpieza
            df_clean = self._run_cleaning_pipeline(df)
            
            if df_clean is None or df_clean.empty:
                if self.verbose:
                    print(f"  ❌ Datos vacíos después de limpieza")
                return None
            
            final_shape = df_clean.shape
            self.cleaning_stats['total_rows_after'] += final_shape[0]
            
            if self.verbose:
                print(f"  ✅ Datos limpios: {final_shape}")
                if final_shape[0] != original_shape[0]:
                    removed = original_shape[0] - final_shape[0]
                    print(f"     Filas eliminadas: {removed}")
            
            self.cleaning_stats['files_successful'] += 1
            return df_clean
            
        except Exception as e:
            if self.verbose:
                print(f"  ❌ Error: {str(e)}")
            self.cleaning_stats['files_failed'] += 1
            return None
        finally:
            self.cleaning_stats['files_processed'] += 1
    
    def _load_csv_safely(self, csv_path: Path) -> Optional[pd.DataFrame]:
        """Carga CSV con manejo robusto de errores."""
        try:
            # Intentar diferentes codificaciones
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(csv_path, encoding=encoding)
                    
                    # Verificar que el DataFrame no esté vacío
                    if df.empty:
                        if self.verbose:
                            print(f"    ⚠️ Archivo vacío: {csv_path.name}")
                        return None
                    
                    # Verificar que tenga al menos 2 columnas
                    if df.shape[1] < 2:
                        if self.verbose:
                            print(f"    ⚠️ Muy pocas columnas ({df.shape[1]}): {csv_path.name}")
                        return None
                    
                    return df
                    
                except UnicodeDecodeError:
                    continue
                except pd.errors.EmptyDataError:
                    if self.verbose:
                        print(f"    ⚠️ Archivo sin datos: {csv_path.name}")
                    return None
                except Exception as e:
                    if encoding == encodings[-1]:  # Último intento
                        if self.verbose:
                            print(f"    ❌ Error cargando {csv_path.name}: {e}")
                        return None
                    continue
            
            return None
            
        except Exception as e:
            if self.verbose:
                print(f"    ❌ Error general cargando {csv_path.name}: {e}")
            return None
    
    def _add_metadata(self, df: pd.DataFrame, csv_path: Path, 
                     class_name: Optional[str], video_id: Optional[str]) -> pd.DataFrame:
        """Agrega metadatos al DataFrame."""
        df_with_meta = df.copy()
        
        # Agregar clase
        if class_name:
            df_with_meta['clase'] = class_name
        elif 'clase' not in df_with_meta.columns:
            # Intentar extraer de la ruta
            possible_classes = ['normal', 'forcejeo', 'sospechoso', 'merodeo']
            for cls in possible_classes:
                if cls.lower() in str(csv_path).lower():
                    df_with_meta['clase'] = cls
                    break
            else:
                df_with_meta['clase'] = 'unknown'
        
        # Agregar video_id
        if video_id is None:
            video_id = csv_path.stem.replace('clean_', '')
        df_with_meta['video_id'] = video_id
        
        return df_with_meta
    
    def _run_cleaning_pipeline(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Ejecuta el pipeline completo de limpieza."""
        try:
            # 1. Limpieza de valores problemáticos
            df_clean = self._clean_problematic_values(df)
            
            # 2. Convertir texto a numérico
            df_clean = self._convert_text_to_numeric(df_clean)
            
            # 3. Manejo de valores faltantes
            df_clean = self._handle_missing_values(df_clean)
            
            # 4. Limpieza de columnas problemáticas
            df_clean = self._clean_problematic_columns(df_clean)
            
            # 5. Validación final
            df_clean = self._final_validation(df_clean)
            
            return df_clean
            
        except Exception as e:
            if self.verbose:
                print(f"    ❌ Error en pipeline de limpieza: {e}")
            return None
    
    def _clean_problematic_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limpia valores infinitos, NaN extremos, y outliers."""
        df_clean = df.copy()
        
        # 1. Reemplazar infinitos con NaN
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        
        # Contar infinitos antes de reemplazar
        inf_count = 0
        for col in numeric_cols:
            inf_mask = np.isinf(df_clean[col])
            inf_count += inf_mask.sum()
        
        if inf_count > 0:
            df_clean[numeric_cols] = df_clean[numeric_cols].replace([np.inf, -np.inf], np.nan)
            self.cleaning_stats['inf_values_replaced'] += inf_count
            if self.verbose:
                print(f"    🔧 Reemplazados {inf_count} valores infinitos")
        
        # 2. Detectar y corregir valores extremos
        protected_cols = ['frame', 'objeto', 'centroide_x', 'centroide_y', 'clase', 'video_id']
        
        for col in numeric_cols:
            if any(protected in col.lower() for protected in protected_cols):
                continue  # No tocar coordenadas y IDs
                
            if df_clean[col].dtype in ['float64', 'int64'] and not df_clean[col].empty:
                # Detectar outliers extremos usando IQR robusto
                outliers_cleaned = self._handle_outliers(df_clean[col], col)
                if outliers_cleaned is not None:
                    df_clean[col] = outliers_cleaned
        
        return df_clean
    
    def _handle_outliers(self, series: pd.Series, col_name: str) -> Optional[pd.Series]:
        """Maneja outliers de manera inteligente."""
        try:
            # Solo procesar si hay suficientes datos válidos
            valid_data = series.dropna()
            if len(valid_data) < 10:
                return series
            
            Q1 = valid_data.quantile(0.25)
            Q3 = valid_data.quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR == 0:  # Datos constantes
                return series
            
            # Límites extremos (más allá de 3*IQR)
            lower_extreme = Q1 - 3 * IQR
            upper_extreme = Q3 + 3 * IQR
            
            # Detectar outliers extremos
            extreme_mask = (series < lower_extreme) | (series > upper_extreme)
            extreme_count = extreme_mask.sum()
            
            if extreme_count > 0 and extreme_count < len(series) * 0.1:  # Máximo 10% de outliers
                if self.verbose:
                    print(f"    🔧 {col_name}: {extreme_count} valores extremos detectados")
                
                # Crear copia para modificar
                series_clean = series.copy()
                
                # Reemplazar con límites razonables en lugar de eliminar
                series_clean.loc[series < lower_extreme] = lower_extreme
                series_clean.loc[series > upper_extreme] = upper_extreme
                
                return series_clean
            
            return series
            
        except Exception as e:
            if self.verbose:
                print(f"    ⚠️ Error manejando outliers en {col_name}: {e}")
            return series
    
    def _convert_text_to_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convierte características de texto a numéricas."""
        df_clean = df.copy()
        
        # Identificar columnas de texto
        text_cols = df_clean.select_dtypes(include=['object']).columns
        exclude_cols = ['clase', 'video_id']  # No convertir metadatos
        
        convertible_cols = [col for col in text_cols if col not in exclude_cols]
        
        if len(convertible_cols) > 0 and self.verbose:
            print(f"    🔤 Convirtiendo {len(convertible_cols)} columnas de texto a numérico")
        
        for col in convertible_cols:
            try:
                converted_col = self._convert_single_text_column(df_clean[col], col)
                if converted_col is not None:
                    df_clean[col] = converted_col
                    self.cleaning_stats['text_columns_converted'] += 1
                    
            except Exception as e:
                if self.verbose:
                    print(f"      ❌ Error convirtiendo {col}: {e}")
        
        return df_clean
    
    def _convert_single_text_column(self, series: pd.Series, col_name: str) -> Optional[pd.Series]:
        """Convierte una sola columna de texto a numérica."""
        # Obtener valores únicos no nulos
        unique_values = series.dropna().unique()
        
        if len(unique_values) == 0:
            return pd.to_numeric(series, errors='coerce')
        
        if self.verbose:
            print(f"      • {col_name} ({len(unique_values)} valores únicos)")
        
        # Intentar conversión directa primero
        numeric_series = pd.to_numeric(series, errors='coerce')
        if not numeric_series.isna().all():
            # Si algún valor se pudo convertir, usar esa conversión
            return numeric_series
        
        if len(unique_values) <= 50:  # Pocas categorías - usar LabelEncoder
            try:
                # Crear encoder si no existe
                if col_name not in self.label_encoders:
                    self.label_encoders[col_name] = LabelEncoder()
                
                # Manejar valores NaN
                mask_not_null = series.notna()
                
                if mask_not_null.any():
                    # Ajustar encoder solo con valores no nulos
                    non_null_values = series.loc[mask_not_null]
                    self.label_encoders[col_name].fit(non_null_values)
                    
                    # Crear serie resultado
                    result_series = pd.Series(np.nan, index=series.index, dtype='float64')
                    
                    # Transformar valores no nulos
                    encoded_values = self.label_encoders[col_name].transform(non_null_values)
                    result_series.loc[mask_not_null] = encoded_values
                    
                    if self.verbose:
                        mapping = dict(zip(
                            self.label_encoders[col_name].classes_, 
                            range(len(self.label_encoders[col_name].classes_))
                        ))
                        print(f"        → Codificado: {mapping}")
                    
                    return result_series
                
            except Exception as e:
                if self.verbose:
                    print(f"        ❌ Error en LabelEncoder: {e}")
        
        else:  # Muchas categorías - usar hash
            if self.verbose:
                print(f"        → Demasiadas categorías ({len(unique_values)}), usando hash")
            
            try:
                return series.astype(str).apply(
                    lambda x: hash(str(x)) % 10000 if pd.notna(x) else np.nan
                )
            except Exception as e:
                if self.verbose:
                    print(f"        ❌ Error en hash: {e}")
        
        # Fallback: retornar serie numérica con NaN
        return pd.Series(np.nan, index=series.index, dtype='float64')
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Maneja valores faltantes de forma inteligente."""
        df_clean = df.copy()
        
        # Separar por tipos
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        
        # Excluir metadatos
        exclude_cols = ['clase', 'video_id']
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
        categorical_cols = [col for col in categorical_cols if col not in exclude_cols]
        
        # Manejar columnas numéricas
        nan_filled = 0
        for col in numeric_cols:
            if df_clean[col].isnull().any():
                filled_count = self._fill_numeric_column(df_clean, col)
                nan_filled += filled_count
        
        # Manejar columnas categóricas
        for col in categorical_cols:
            if df_clean[col].isnull().any():
                filled_count = self._fill_categorical_column(df_clean, col)
                nan_filled += filled_count
        
        self.cleaning_stats['nan_values_filled'] += nan_filled
        
        return df_clean
    
    def _fill_numeric_column(self, df: pd.DataFrame, col: str) -> int:
        """Rellena una columna numérica basada en su contexto."""
        null_count = df[col].isnull().sum()
        if null_count == 0:
            return 0
        
        null_percentage = (null_count / len(df)) * 100
        
        if self.verbose:
            print(f"    🔧 {col}: {null_count} NaN ({null_percentage:.1f}%)")
        
        # Estrategia según el tipo de característica
        col_lower = col.lower()
        fill_value = None
        fill_method = "mediana"
        
        # Reglas específicas por tipo de característica
        if any(word in col_lower for word in ['frame', 'tiempo', 'duracion', 'time']):
            # Para tiempo: usar interpolación
            if len(df[col].dropna()) > 1:
                df[col] = df[col].interpolate(method='linear')
                fill_method = "interpolación"
            else:
                df[col] = df[col].fillna(0)
                fill_method = "0"
                
        elif any(word in col_lower for word in ['count', 'numero', 'cantidad', 'num_']):
            # Para conteos: usar 0
            df[col] = df[col].fillna(0)
            fill_method = "0"
            
        elif any(word in col_lower for word in ['velocidad', 'speed', 'vel_']):
            # Para velocidad: usar 0 (objeto parado)
            df[col] = df[col].fillna(0)
            fill_method = "0 (parado)"
            
        elif any(word in col_lower for word in ['aceleracion', 'acceleration', 'acel_']):
            # Para aceleración: usar 0 (velocidad constante)
            df[col] = df[col].fillna(0)
            fill_method = "0 (constante)"
            
        elif any(word in col_lower for word in ['direccion', 'direction', 'angle', 'angulo']):
            # Para dirección: usar la dirección más común o 0
            mode_values = df[col].mode()
            fill_value = mode_values[0] if len(mode_values) > 0 else 0
            df[col] = df[col].fillna(fill_value)
            fill_method = f"moda ({fill_value:.2f})" if fill_value != 0 else "0"
            
        elif any(word in col_lower for word in ['densidad', 'density', 'dens_']):
            # Para densidad: usar mediana
            median_val = df[col].median()
            if pd.notna(median_val):
                df[col] = df[col].fillna(median_val)
                fill_method = f"mediana ({median_val:.2f})"
            else:
                df[col] = df[col].fillna(0)
                fill_method = "0 (mediana era NaN)"
                
        elif any(word in col_lower for word in ['distancia', 'distance', 'dist_']):
            # Para distancias: usar mediana o 0
            median_val = df[col].median()
            if pd.notna(median_val):
                df[col] = df[col].fillna(median_val)
                fill_method = f"mediana ({median_val:.2f})"
            else:
                df[col] = df[col].fillna(0)
                fill_method = "0"
                
        elif any(word in col_lower for word in ['ciclico', 'ciclo', 'cycle']):
            # Para variables cíclicas: usar 0 (no cíclico)
            df[col] = df[col].fillna(0)
            fill_method = "0 (no cíclico)"
            
        elif any(word in col_lower for word in ['interaccion', 'interaction']):
            # Para interacciones: usar 0 (sin interacción)
            df[col] = df[col].fillna(0)
            fill_method = "0 (sin interacción)"
            
        else:
            # Para otros: usar mediana
            median_val = df[col].median()
            if pd.notna(median_val):
                df[col] = df[col].fillna(median_val)
                fill_method = f"mediana ({median_val:.2f})"
            else:
                df[col] = df[col].fillna(0)
                fill_method = "0 (mediana era NaN)"
        
        if self.verbose:
            print(f"        → Rellenado con: {fill_method}")
        
        return null_count
    
    def _fill_categorical_column(self, df: pd.DataFrame, col: str) -> int:
        """Rellena una columna categórica."""
        null_count = df[col].isnull().sum()
        if null_count == 0:
            return 0
        
        mode_values = df[col].mode()
        
        if len(mode_values) > 0:
            df[col] = df[col].fillna(mode_values[0])
            if self.verbose:
                print(f"    🔤 {col}: {null_count} NaN rellenados con moda '{mode_values[0]}'")
        else:
            df[col] = df[col].fillna('unknown')
            if self.verbose:
                print(f"    🔤 {col}: {null_count} NaN rellenados con 'unknown'")
        
        return null_count
    
    def _clean_problematic_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limpia columnas que pueden causar problemas."""
        df_clean = df.copy()
        
        # Eliminar columnas completamente vacías
        empty_cols = df_clean.columns[df_clean.isnull().all()].tolist()
        if empty_cols:
            df_clean = df_clean.drop(columns=empty_cols)
            if self.verbose:
                print(f"    🗑️ Eliminadas {len(empty_cols)} columnas vacías: {empty_cols}")
        
        # Eliminar columnas constantes (excepto metadatos)
        exclude_from_constant_check = ['clase', 'video_id']
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        cols_to_check = [col for col in numeric_cols if col not in exclude_from_constant_check]
        
        constant_cols = []
        for col in cols_to_check:
            if df_clean[col].nunique() <= 1:
                constant_cols.append(col)
        
        if constant_cols:
            df_clean = df_clean.drop(columns=constant_cols)
            if self.verbose:
                print(f"    🗑️ Eliminadas {len(constant_cols)} columnas constantes")
        
        return df_clean
    
    def _final_validation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validación final y limpieza de seguridad."""
        df_clean = df.copy()
        
        # 1. Verificar que no queden NaN
        remaining_nan = df_clean.isnull().sum().sum()
        if remaining_nan > 0:
            if self.verbose:
                print(f"    ⚠️ Quedan {remaining_nan} NaN - limpieza final")
            
            # Limpieza agresiva final
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            categorical_cols = df_clean.select_dtypes(include=['object']).columns
            
            exclude_cols = ['clase', 'video_id']
            numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
            categorical_cols = [col for col in categorical_cols if col not in exclude_cols]
            
            # Rellenar NaN restantes
            for col in numeric_cols:
                if df_clean[col].isnull().any():
                    df_clean[col] = df_clean[col].fillna(0)
            
            for col in categorical_cols:
                if df_clean[col].isnull().any():
                    df_clean[col] = df_clean[col].fillna('unknown')
        
        # 2. Verificar infinitos
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            inf_check = np.isinf(df_clean[numeric_cols]).sum().sum()
            if inf_check > 0:
                if self.verbose:
                    print(f"    🚨 Encontrados {inf_check} valores infinitos - corrigiendo")
                df_clean[numeric_cols] = df_clean[numeric_cols].replace([np.inf, -np.inf], 0)
        
        # 3. Eliminar filas completamente vacías (excepto metadatos)
        initial_rows = len(df_clean)
        
        # Identificar filas donde todas las columnas (excepto metadatos) son NaN
        exclude_for_empty_check = ['clase', 'video_id']
        data_cols = [col for col in df_clean.columns if col not in exclude_for_empty_check]
        
        if data_cols:
            empty_data_mask = df_clean[data_cols].isnull().all(axis=1)
            df_clean = df_clean[~empty_data_mask]
        
        final_rows = len(df_clean)
        
        if initial_rows != final_rows and self.verbose:
            print(f"    🗑️ Eliminadas {initial_rows - final_rows} filas sin datos útiles")
        
        # 4. Verificación final
        if len(df_clean) > 0:
            final_nan = df_clean.isnull().sum().sum()
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            final_inf = np.isinf(df_clean[numeric_cols]).sum().sum() if len(numeric_cols) > 0 else 0
            
            if final_nan > 0 or final_inf > 0:
                if self.verbose:
                    print(f"    ❌ ADVERTENCIA: Quedan {final_nan} NaN y {final_inf} infinitos")
            else:
                if self.verbose:
                    print(f"    ✅ Validación final exitosa")
        
        return df_clean
    
    def clean_directory(self, csv_dir: str = '../../csv', 
                       output_dir: str = './cleaned_data/') -> Tuple[List[pd.DataFrame], Dict[str, Any]]:
        """
        Limpia todos los CSVs en un directorio organizado por clases.
        
        Parámetros:
        -----------
        csv_dir : str
            Directorio con CSVs organizados por clase
        output_dir : str
            Directorio para guardar CSVs limpios
        """
        csv_dir = Path(csv_dir)
        output_dir = Path(output_dir)
        
        if self.verbose:
            print("🚀 LIMPIEZA COMPLETA DE DIRECTORIO")
            print("=" * 50)
        
        if not csv_dir.exists():
            if self.verbose:
                print(f"❌ Directorio no encontrado: {csv_dir}")
            return [], {}
        
        # Crear directorio de salida
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Buscar clases disponibles
        classes = self._find_available_classes(csv_dir)
        
        if not classes and self.verbose:
            print("⚠️ No se encontraron clases organizadas, procesando archivos directamente")
            return self._clean_unorganized_directory(csv_dir, output_dir)
        
        all_cleaned_data = []
        cleaning_summary = {
            'total_files': 0,
            'successful': 0,
            'failed': 0,
            'classes': {},
            'cleaning_stats': self.cleaning_stats.copy()
        }
        
        # Procesar cada clase
        for class_name in classes:
            class_data = self._clean_class_directory(csv_dir, output_dir, class_name)
            
            if class_data['cleaned_dataframes']:
                all_cleaned_data.extend(class_data['cleaned_dataframes'])
            
            cleaning_summary['classes'][class_name] = class_data['summary']
            cleaning_summary['total_files'] += class_data['summary']['files']
            cleaning_summary['successful'] += class_data['summary']['successful']
            cleaning_summary['failed'] += class_data['summary']['failed']
        
        # Combinar y guardar todos los datos limpios
        if all_cleaned_data:
            self._save_combined_data(all_cleaned_data, output_dir)
        
        # Generar reporte
        self._generate_cleaning_report(cleaning_summary, output_dir)
        
        # Actualizar estadísticas finales
        cleaning_summary['cleaning_stats'] = self.cleaning_stats
        
        return all_cleaned_data, cleaning_summary
    
    def _find_available_classes(self, csv_dir: Path) -> List[str]:
        """Encuentra las clases disponibles en el directorio."""
        possible_classes = ['normal', 'forcejeo', 'sospechoso', 'merodeo']
        available_classes = []
        
        for class_name in possible_classes:
            class_dir = csv_dir / class_name
            if class_dir.exists() and class_dir.is_dir():
                # Verificar que contenga archivos CSV
                csv_files = list(class_dir.glob("*.csv"))
                if csv_files:
                    available_classes.append(class_name)
        
        return available_classes
    
    def _clean_unorganized_directory(self, csv_dir: Path, output_dir: Path) -> Tuple[List[pd.DataFrame], Dict[str, Any]]:
        """Limpia directorio sin organización por clases."""
        csv_files = list(csv_dir.glob("*.csv"))
        
        if not csv_files:
            if self.verbose:
                print("❌ No se encontraron archivos CSV")
            return [], {}
        
        if self.verbose:
            print(f"📁 Procesando {len(csv_files)} archivos CSV sin clasificar")
        
        all_cleaned_data = []
        successful = 0
        failed = 0
        
        for csv_file in csv_files:
            cleaned_df = self.clean_single_csv(csv_file)
            
            if cleaned_df is not None:
                # Guardar archivo limpio
                output_path = output_dir / f"clean_{csv_file.name}"
                cleaned_df.to_csv(output_path, index=False)
                all_cleaned_data.append(cleaned_df)
                successful += 1
            else:
                failed += 1
        
        summary = {
            'total_files': len(csv_files),
            'successful': successful,
            'failed': failed,
            'classes': {'unorganized': {'files': len(csv_files), 'successful': successful, 'failed': failed}},
            'cleaning_stats': self.cleaning_stats
        }
        
        return all_cleaned_data, summary
    
    def _clean_class_directory(self, csv_dir: Path, output_dir: Path, class_name: str) -> Dict[str, Any]:
        """Limpia archivos de una clase específica."""
        class_dir = csv_dir / class_name
        class_output_dir = output_dir / class_name
        class_output_dir.mkdir(parents=True, exist_ok=True)
        
        csv_files = list(class_dir.glob("*.csv"))
        
        if self.verbose:
            print(f"\n📁 Procesando clase '{class_name}': {len(csv_files)} archivos")
        
        cleaned_dataframes = []
        successful = 0
        failed = 0
        
        for csv_file in csv_files:
            output_path = class_output_dir / f"clean_{csv_file.name}"
            
            # Limpiar archivo
            cleaned_df = self.clean_single_csv(csv_file, class_name)
            
            if cleaned_df is not None:
                # Guardar archivo limpio
                try:
                    cleaned_df.to_csv(output_path, index=False)
                    cleaned_dataframes.append(cleaned_df)
                    successful += 1
                except Exception as e:
                    if self.verbose:
                        print(f"  ❌ Error guardando {output_path}: {e}")
                    failed += 1
            else:
                failed += 1
        
        if self.verbose:
            print(f"  ✅ {class_name}: {successful}/{len(csv_files)} exitosos")
        
        return {
            'cleaned_dataframes': cleaned_dataframes,
            'summary': {
                'files': len(csv_files),
                'successful': successful,
                'failed': failed
            }
        }
    
    def _save_combined_data(self, all_cleaned_data: List[pd.DataFrame], output_dir: Path):
        """Guarda datos combinados y genera estadísticas."""
        try:
            combined_clean = pd.concat(all_cleaned_data, ignore_index=True)
            combined_path = output_dir / 'all_data_cleaned.csv'
            combined_clean.to_csv(combined_path, index=False)
            
            if self.verbose:
                print(f"\n📊 DATOS COMBINADOS GUARDADOS:")
                print(f"   Archivo: {combined_path}")
                print(f"   Total filas: {len(combined_clean):,}")
                print(f"   Columnas: {len(combined_clean.columns)}")
                
                if 'clase' in combined_clean.columns:
                    print(f"   Distribución por clase:")
                    class_counts = combined_clean['clase'].value_counts()
                    for clase, count in class_counts.items():
                        percentage = (count / len(combined_clean)) * 100
                        print(f"     {clase}: {count:,} ({percentage:.1f}%)")
        
        except Exception as e:
            if self.verbose:
                print(f"❌ Error guardando datos combinados: {e}")
    
    def _generate_cleaning_report(self, summary: Dict[str, Any], output_dir: Path):
        """Genera reporte detallado de limpieza."""
        try:
            report_path = output_dir / 'cleaning_report.txt'
            
            # Calcular estadísticas
            total_files = summary['total_files']
            successful = summary['successful']
            failed = summary['failed']
            success_rate = (successful / total_files * 100) if total_files > 0 else 0
            
            # Contenido del reporte
            report_content = f"""
REPORTE DE LIMPIEZA DE DATOS
============================

Fecha: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

RESUMEN GENERAL:
- Total de archivos procesados: {total_files:,}
- Archivos exitosos: {successful:,}
- Archivos fallidos: {failed:,}
- Tasa de éxito: {success_rate:.1f}%

ESTADÍSTICAS DE LIMPIEZA:
- Filas procesadas (antes): {self.cleaning_stats['total_rows_before']:,}
- Filas finales (después): {self.cleaning_stats['total_rows_after']:,}
- Valores NaN rellenados: {self.cleaning_stats['nan_values_filled']:,}
- Valores infinitos reemplazados: {self.cleaning_stats['inf_values_replaced']:,}
- Columnas de texto convertidas: {self.cleaning_stats['text_columns_converted']:,}

DETALLES POR CLASE:
"""
            
            for class_name, class_data in summary['classes'].items():
                files = class_data['files']
                successful_class = class_data['successful']
                failed_class = class_data['failed']
                success_rate_class = (successful_class / files * 100) if files > 0 else 0
                
                report_content += f"""
{class_name.upper()}:
  - Archivos: {files:,}
  - Exitosos: {successful_class:,}
  - Fallidos: {failed_class:,}
  - Tasa de éxito: {success_rate_class:.1f}%
"""
            
            report_content += f"""

TRANSFORMACIONES APLICADAS:
1. ✅ Carga segura con múltiples codificaciones
2. ✅ Valores infinitos → NaN → Rellenados inteligentemente
3. ✅ Outliers extremos → Limitados a rangos razonables (IQR)
4. ✅ Texto → Numérico (LabelEncoder para ≤50 categorías, Hash para >50)
5. ✅ NaN → Rellenados según contexto semántico:
   - Velocidades/Aceleraciones → 0 (estado de reposo)
   - Conteos/Cantidades → 0
   - Direcciones/Ángulos → Moda
   - Distancias/Densidades → Mediana
   - Tiempo/Duración → Interpolación lineal
   - Interacciones → 0 (sin interacción)
   - Otros → Mediana
6. ✅ Eliminación de columnas vacías y constantes
7. ✅ Validación final sin NaN ni infinitos

CODIFICADORES UTILIZADOS:
"""
            
            if self.label_encoders:
                for col, encoder in self.label_encoders.items():
                    try:
                        mappings = dict(zip(encoder.classes_, range(len(encoder.classes_))))
                        # Limitar la longitud del mapeo para legibilidad
                        if len(mappings) > 10:
                            sample_mappings = dict(list(mappings.items())[:5])
                            report_content += f"  {col}: {sample_mappings}... (total: {len(mappings)} categorías)\n"
                        else:
                            report_content += f"  {col}: {mappings}\n"
                    except Exception as e:
                        report_content += f"  {col}: Error mostrando mapeo - {e}\n"
            else:
                report_content += "  No se utilizaron codificadores de texto\n"
            
            report_content += f"""

CALIDAD DE DATOS FINAL:
- Archivos procesados exitosamente: {successful}/{total_files}
- Consistencia de datos: {'✅ Excelente' if success_rate > 90 else '⚠️ Revisar' if success_rate > 70 else '❌ Problemas'}
- Datos listos para Machine Learning: {'✅ Sí' if successful > 0 else '❌ No'}

RECOMENDACIONES:
"""
            
            if success_rate < 70:
                report_content += "- ⚠️ Baja tasa de éxito. Revisar formato de archivos de entrada\n"
            if failed > 0:
                report_content += f"- 🔍 Investigar {failed} archivos fallidos para mejorar robustez\n"
            if self.cleaning_stats['inf_values_replaced'] > 1000:
                report_content += "- 📊 Muchos valores infinitos detectados. Revisar cálculos de características\n"
            if successful > 0:
                report_content += "- ✅ Datos listos para el siguiente paso: análisis de características\n"
            
            # Guardar reporte
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            if self.verbose:
                print(f"\n📄 Reporte detallado guardado: {report_path}")
        
        except Exception as e:
            if self.verbose:
                print(f"⚠️ Error generando reporte: {e}")
    
    def get_cleaning_statistics(self) -> Dict[str, Any]:
        """Retorna estadísticas detalladas de limpieza."""
        return {
            'processing_stats': self.cleaning_stats.copy(),
            'encoders_used': len(self.label_encoders),
            'encoder_mappings': {
                col: dict(zip(encoder.classes_, range(len(encoder.classes_))))
                for col, encoder in self.label_encoders.items()
            }
        }
    
    def save_encoders(self, output_dir: str):
        """Guarda los encoders para uso futuro."""
        output_dir = Path(output_dir)
        encoders_file = output_dir / 'label_encoders.json'
        
        try:
            encoders_data = {}
            for col, encoder in self.label_encoders.items():
                encoders_data[col] = {
                    'classes': encoder.classes_.tolist(),
                    'mapping': dict(zip(encoder.classes_, range(len(encoder.classes_))))
                }
            
            with open(encoders_file, 'w', encoding='utf-8') as f:
                json.dump(encoders_data, f, indent=2, ensure_ascii=False)
            
            if self.verbose:
                print(f"💾 Codificadores guardados: {encoders_file}")
        
        except Exception as e:
            if self.verbose:
                print(f"⚠️ Error guardando codificadores: {e}")
    
    def load_encoders(self, encoders_file: str):
        """Carga encoders previamente guardados."""
        encoders_file = Path(encoders_file)
        
        if not encoders_file.exists():
            if self.verbose:
                print(f"⚠️ Archivo de codificadores no encontrado: {encoders_file}")
            return
        
        try:
            with open(encoders_file, 'r', encoding='utf-8') as f:
                encoders_data = json.load(f)
            
            for col, encoder_info in encoders_data.items():
                self.label_encoders[col] = LabelEncoder()
                self.label_encoders[col].classes_ = np.array(encoder_info['classes'])
            
            if self.verbose:
                print(f"📥 {len(encoders_data)} codificadores cargados desde: {encoders_file}")
        
        except Exception as e:
            if self.verbose:
                print(f"❌ Error cargando codificadores: {e}")

def main():
    """
    Función principal para ejecutar la limpieza interactiva.
    """
    print("🧹 LIMPIADOR COMPLETO DE DATOS")
    print("=" * 50)
    print("🎯 Este script:")
    print("   • Carga CSVs con múltiples codificaciones")
    print("   • Limpia valores infinitos y NaN inteligentemente")
    print("   • Convierte texto a numérico automáticamente")
    print("   • Maneja outliers extremos de forma robusta")
    print("   • Genera reporte detallado con estadísticas")
    print("   • Guarda datos limpios organizados por clase")
    
    # Configuración interactiva
    print("\n📋 CONFIGURACIÓN:")
    
    # Directorio de entrada
    default_input = "../../csv"
    csv_directory = input(f"📁 Directorio con CSVs [{default_input}]: ").strip()
    if not csv_directory:
        csv_directory = default_input
    
    # Directorio de salida
    default_output = "./cleaned_data/"
    output_directory = input(f"💾 Directorio de salida [{default_output}]: ").strip()
    if not output_directory:
        output_directory = default_output
    
    # Verificar directorio de entrada
    if not os.path.exists(csv_directory):
        print(f"❌ Error: {csv_directory} no existe")
        
        # Ofrecer alternativas
        print("\n🔍 Buscando directorios alternativos...")
        current_dir = Path(".")
        csv_dirs = [d for d in current_dir.rglob("*") if d.is_dir() and "csv" in d.name.lower()]
        
        if csv_dirs:
            print("📁 Directorios encontrados con 'csv' en el nombre:")
            for i, dir_path in enumerate(csv_dirs[:5], 1):
                print(f"   {i}. {dir_path}")
            
            choice = input("Seleccionar directorio (número) o 'q' para salir: ").strip()
            if choice.isdigit() and 1 <= int(choice) <= min(5, len(csv_dirs)):
                csv_directory = str(csv_dirs[int(choice) - 1])
            else:
                print("👋 Saliendo...")
                return
        else:
            print("❌ No se encontraron directorios alternativos")
            return
    
    # Opciones avanzadas
    print("\n⚙️ OPCIONES AVANZADAS:")
    verbose = input("🔊 Mostrar información detallada [S/n]: ").strip().lower() not in ['n', 'no']
    save_encoders_option = input("💾 Guardar codificadores para reutilizar [S/n]: ").strip().lower() not in ['n', 'no']
    
    print(f"\n🚀 INICIANDO LIMPIEZA...")
    print(f"   Entrada: {csv_directory}")
    print(f"   Salida: {output_directory}")
    print(f"   Verbose: {'✅' if verbose else '❌'}")
    
    # Ejecutar limpieza
    try:
        cleaner = ComprehensiveDataCleaner(verbose=verbose)
        cleaned_data, summary = cleaner.clean_directory(csv_directory, output_directory)
        
        # Guardar codificadores si se solicita
        if save_encoders_option and cleaner.label_encoders:
            cleaner.save_encoders(output_directory)
        
        # Mostrar resumen final
        print(f"\n🎉 LIMPIEZA COMPLETADA")
        print(f"   Archivos procesados: {summary['total_files']:,}")
        print(f"   Exitosos: {summary['successful']:,}")
        print(f"   Fallidos: {summary['failed']:,}")
        print(f"   Datos limpios en: {output_directory}")
        
        # Estadísticas de transformación
        stats = cleaner.get_cleaning_statistics()
        print(f"\n📊 ESTADÍSTICAS DE TRANSFORMACIÓN:")
        print(f"   Filas procesadas: {stats['processing_stats']['total_rows_before']:,}")
        print(f"   Filas finales: {stats['processing_stats']['total_rows_after']:,}")
        print(f"   NaN rellenados: {stats['processing_stats']['nan_values_filled']:,}")
        print(f"   Infinitos corregidos: {stats['processing_stats']['inf_values_replaced']:,}")
        print(f"   Columnas texto convertidas: {stats['processing_stats']['text_columns_converted']:,}")
        
        # Mostrar ejemplo de uso
        if cleaned_data:
            print(f"\n💡 EJEMPLO DE USO DE DATOS LIMPIOS:")
            print(f"""
# Para usar los datos limpios:
import pandas as pd

# Cargar datos combinados
df = pd.read_csv('{output_directory}/all_data_cleaned.csv')
print(f"Datos cargados: {{df.shape}}")
print(f"Clases disponibles: {{df['clase'].unique()}}")

# O cargar por clase específica:
# df_normal = pd.read_csv('{output_directory}/normal/clean_ejemplo.csv')

# Los datos ya están listos para ML:
# - Sin NaN ni infinitos ✅
# - Texto convertido a numérico ✅  
# - Outliers controlados ✅
# - Metadatos incluidos (clase, video_id) ✅

# Próximo paso sugerido:
# python enhanced_data_processor.py
""")
        
        # Recomendaciones
        success_rate = (summary['successful'] / summary['total_files'] * 100) if summary['total_files'] > 0 else 0
        
        print(f"\n🎯 RECOMENDACIONES:")
        if success_rate >= 90:
            print("   ✅ Excelente calidad de datos - listos para procesamiento")
        elif success_rate >= 70:
            print("   ⚠️ Buena calidad, pero revisar archivos fallidos")
        else:
            print("   ❌ Revisar formato de archivos de entrada")
        
        if summary['failed'] > 0:
            print(f"   🔍 Investigar {summary['failed']} archivos fallidos")
        
        print("   📈 Siguiente paso: Ejecutar enhanced_data_processor.py")
        print("   📊 Para análisis: Ejecutar enhanced_feature_analyzer.py")
    
    except KeyboardInterrupt:
        print(f"\n⏹️ Proceso interrumpido por el usuario")
    except Exception as e:
        print(f"\n❌ Error durante la limpieza: {e}")
        import traceback
        traceback.print_exc()

def quick_clean(input_dir: str, output_dir: str, verbose: bool = True) -> Tuple[List[pd.DataFrame], Dict[str, Any]]:
    """
    Función de limpieza rápida para uso programático.
    
    Parámetros:
    -----------
    input_dir : str
        Directorio con archivos CSV a limpiar
    output_dir : str
        Directorio donde guardar archivos limpios
    verbose : bool
        Si mostrar información detallada
    
    Retorna:
    --------
    Tuple[List[pd.DataFrame], Dict[str, Any]]
        Lista de DataFrames limpios y diccionario con estadísticas
    """
    cleaner = ComprehensiveDataCleaner(verbose=verbose)
    return cleaner.clean_directory(input_dir, output_dir)

if __name__ == "__main__":
    main()