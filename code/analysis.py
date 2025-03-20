"""
Módulo para analizar datos de características de movimiento.
Este script procesa los archivos CSV generados por el sistema de detección de objetos
y extracción de características para identificar patrones de comportamiento sospechoso.
"""
import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

class MovementAnalyzer:
    def __init__(self, csv_directory):
        """
        Inicializa el analizador de movimientos
        
        Parámetros:
        -----------
        csv_directory : str
            Ruta al directorio que contiene los archivos CSV con datos de movimiento
        """
        self.csv_directory = csv_directory
        self.dataframes = {}  # Almacena DataFrames por video
        self.anomalias = {}   # Almacena resultados de detección de anomalías
        self.combined_data = None  # Para análisis combinado de todos los videos
        
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
        
        # Actualizar también los DataFrames individuales
        for video_id, df in self.dataframes.items():
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            for col in df.select_dtypes(include=['float64', 'int64']).columns:
                df[col].fillna(df[col].median(), inplace=True)
            for col in df.select_dtypes(include=['object']).columns:
                df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Desconocido", inplace=True)
    
    def calcular_estadisticas_basicas(self):
        """Calcula estadísticas básicas de los datos de movimiento"""
        if self.combined_data is None:
            print("No hay datos cargados para analizar")
            return None
        
        print("\nEstadísticas básicas de movimiento:")
        
        # Columnas numéricas relevantes para estadísticas
        columnas_metricas = [
            'Desplazamiento', 'Velocidad', 'Aceleracion', 
            'Linealidad', 'Circularidad', 'Zigzag',
            'Frecuencia_Ciclo', 'Amplitud_Ciclo', 'Area_Trayectoria'
        ]
        
        # Estadísticas por video
        stats_por_video = {}
        
        for video_id, df in self.dataframes.items():
            stats = df[columnas_metricas].describe()
            stats_por_video[video_id] = stats
            
            print(f"\nVideo: {video_id}")
            print(f"  Duración (frames): {df['Frame'].max() - df['Frame'].min() + 1}")
            print(f"  Objetos detectados: {df['Objeto'].nunique()}")
            print(f"  Velocidad media: {stats.loc['mean', 'Velocidad']:.2f} px/s")
            print(f"  Velocidad máxima: {stats.loc['max', 'Velocidad']:.2f} px/s")
            
            # Patrones de movimiento dominantes
            patrones = df['Patron_Movimiento'].value_counts(normalize=True) * 100
            print("  Patrones de movimiento:")
            for patron, porcentaje in patrones.items():
                print(f"    - {patron}: {porcentaje:.1f}%")
                
            # Proporción de movimientos cíclicos
            ciclicos = df['Es_Ciclico'].mean() * 100
            print(f"  Movimientos cíclicos: {ciclicos:.1f}%")
            
            # Proporción de interacciones
            interacciones = df['En_Interaccion'].mean() * 100
            print(f"  Objetos en interacción: {interacciones:.1f}%")
        
        return stats_por_video
    
    def detectar_anomalias(self, contamination=0.05):
        """
        Detecta movimientos anómalos usando Isolation Forest
        
        Parámetros:
        -----------
        contamination : float
            Proporción esperada de anomalías en los datos
        """
        if self.combined_data is None:
            print("No hay datos cargados para analizar")
            return
        
        print("\nDetectando patrones de movimiento anómalos...")
        
        # Características relevantes para detección de anomalías
        caracteristicas = [
            'Desplazamiento', 'Velocidad', 'Aceleracion', 'Direccion',
            'Linealidad', 'Circularidad', 'Zigzag', 
            'Es_Ciclico', 'Frecuencia_Ciclo', 'Amplitud_Ciclo',
            'Area_Trayectoria', 'En_Interaccion'
        ]
        
        # Analizar cada video por separado
        for video_id, df in self.dataframes.items():
            print(f"\nAnalizando video: {video_id}")
            
            if len(df) < 10:
                print(f"  ⚠ Insuficientes datos para análisis de anomalías")
                continue
                
            # Extraer características numéricas
            X = df[caracteristicas].select_dtypes(include=['float64', 'int64'])
            
            # Estandarizar datos
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Aplicar Isolation Forest
            modelo = IsolationForest(contamination=contamination, random_state=42)
            df['anomalia'] = modelo.fit_predict(X_scaled)
            
            # Convertir a etiquetas más legibles (-1: anomalía, 1: normal)
            df['es_anomalia'] = df['anomalia'].map({1: 0, -1: 1})
            
            # Guardar resultados
            self.anomalias[video_id] = df[df['es_anomalia'] == 1].copy()
            
            # Estadísticas de anomalías
            total_anomalias = df['es_anomalia'].sum()
            porcentaje = (total_anomalias / len(df)) * 100
            print(f"  Anomalías detectadas: {total_anomalias} ({porcentaje:.1f}%)")
            
            if total_anomalias > 0:
                # Características distintivas de las anomalías
                print("  Características de movimientos anómalos:")
                anomalias_df = df[df['es_anomalia'] == 1]
                normal_df = df[df['es_anomalia'] == 0]
                
                for col in X.columns:
                    anomalia_media = anomalias_df[col].mean()
                    normal_media = normal_df[col].mean()
                    diferencia = ((anomalia_media - normal_media) / normal_media) * 100 if normal_media != 0 else float('inf')
                    
                    if abs(diferencia) > 30:  # Solo mostrar diferencias significativas
                        direccion = "mayor" if diferencia > 0 else "menor"
                        print(f"    - {col}: {abs(diferencia):.1f}% {direccion}")
                        
                # Frames con más anomalías
                frames_anomalos = anomalias_df['Frame'].value_counts().sort_values(ascending=False).head(5)
                print("  Frames con más anomalías:")
                for frame, count in frames_anomalos.items():
                    print(f"    - Frame {frame}: {count} anomalías")
    
    def analisis_clustering(self, n_clusters=3):
        """
        Realiza análisis de clustering para identificar patrones de movimiento
        
        Parámetros:
        -----------
        n_clusters : int
            Número de clusters para K-Means
        """
        if self.combined_data is None:
            print("No hay datos cargados para analizar")
            return
        
        print("\nRealizando análisis de clustering de patrones de movimiento...")
        
        # Características para clustering
        features = [
            'Velocidad', 'Direccion', 'Linealidad', 'Circularidad', 
            'Zigzag', 'Es_Ciclico', 'Area_Trayectoria'
        ]
        
        # Preparar datos
        X = self.combined_data[features].select_dtypes(include=['float64', 'int64'])
        
        # Manejar valores NaN remanentes si existen
        X.fillna(X.median(), inplace=True)
        
        # Estandarizar
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Reducir dimensionalidad para visualización
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # Aplicar K-Means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.combined_data['cluster'] = kmeans.fit_predict(X_scaled)
        
        # Añadir componentes principales al DataFrame
        self.combined_data['pca1'] = X_pca[:, 0]
        self.combined_data['pca2'] = X_pca[:, 1]
        
        # Características de los clusters
        print("\nCaracterísticas de los clusters:")
        for i in range(n_clusters):
            cluster_data = self.combined_data[self.combined_data['cluster'] == i]
            print(f"\nCluster {i} - {len(cluster_data)} registros ({len(cluster_data)/len(self.combined_data)*100:.1f}%)")
            
            # Estadísticas del cluster
            for feature in features:
                avg_value = cluster_data[feature].mean()
                total_avg = self.combined_data[feature].mean()
                diff = ((avg_value - total_avg) / total_avg) * 100 if total_avg != 0 else 0
                signo = "+" if diff > 0 else ""
                print(f"  {feature}: {avg_value:.2f} ({signo}{diff:.1f}% vs. promedio)")
            
            # Patrón de movimiento dominante
            if 'Patron_Movimiento' in self.combined_data.columns:
                patron_dominante = cluster_data['Patron_Movimiento'].mode()[0]
                prop = cluster_data['Patron_Movimiento'].value_counts(normalize=True).max() * 100
                print(f"  Patrón dominante: {patron_dominante} ({prop:.1f}%)")
            
            # Videos en los que aparece este cluster
            videos_cluster = cluster_data['Video'].value_counts(normalize=True) * 100
            print("  Distribución por videos:")
            for video, porcentaje in videos_cluster.head(3).items():
                print(f"    - {video}: {porcentaje:.1f}%")
        
        return X_pca
    
    def visualizar_resultados(self):
        """Genera visualizaciones de los datos analizados"""
        if self.combined_data is None:
            print("No hay datos cargados para visualizar")
            return
        
        print("\nGenerando visualizaciones...")
        
        # Configuración estética
        plt.style.use('ggplot')
        sns.set(font_scale=1.1)
        
        # 1. Distribución de velocidades por video
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='Video', y='Velocidad', data=self.combined_data)
        plt.title('Distribución de Velocidades por Video')
        plt.ylabel('Velocidad (píxeles/segundo)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('velocidades_por_video.png')
        print("  ✓ Gráfico de velocidades guardado")
        
        # 2. Mapa de calor de correlaciones
        plt.figure(figsize=(14, 10))
        features_corr = [
            'Desplazamiento', 'Velocidad', 'Aceleracion', 'Direccion',
            'Linealidad', 'Circularidad', 'Zigzag', 'Es_Ciclico',
            'Area_Trayectoria', 'En_Interaccion'
        ]
        corr = self.combined_data[features_corr].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                   vmin=-1, vmax=1, center=0, square=True, linewidths=.5)
        plt.title('Correlaciones entre Características de Movimiento')
        plt.tight_layout()
        plt.savefig('correlaciones_caracteristicas.png')
        print("  ✓ Mapa de correlaciones guardado")
        
        # 3. Visualización de clusters
        if 'cluster' in self.combined_data.columns and 'pca1' in self.combined_data.columns:
            plt.figure(figsize=(12, 8))
            scatter = sns.scatterplot(
                x='pca1', y='pca2', 
                hue='cluster', 
                palette='viridis',
                s=50, alpha=0.7,
                data=self.combined_data
            )
            plt.title('Clustering de Patrones de Movimiento (PCA)')
            plt.xlabel('Componente Principal 1')
            plt.ylabel('Componente Principal 2')
            plt.legend(title='Cluster')
            
            # Añadir centros de cluster
            centers = self.combined_data.groupby('cluster')[['pca1', 'pca2']].mean().values
            plt.scatter(centers[:, 0], centers[:, 1], c='red', s=100, alpha=0.8, marker='X')
            
            plt.tight_layout()
            plt.savefig('clusters_movimiento.png')
            print("  ✓ Visualización de clusters guardada")
        
        # 4. Distribución de anomalías por video (si hay)
        if any(len(df) > 0 for df in self.anomalias.values()):
            anomalias_por_video = {}
            total_por_video = {}
            
            for video_id, df in self.dataframes.items():
                if video_id in self.anomalias:
                    anomalias_por_video[video_id] = len(self.anomalias[video_id])
                    total_por_video[video_id] = len(df)
            
            if anomalias_por_video:
                plt.figure(figsize=(12, 6))
                
                videos = list(anomalias_por_video.keys())
                porcentajes = [anomalias_por_video[v]/total_por_video[v]*100 for v in videos]
                
                sns.barplot(x=videos, y=porcentajes)
                plt.title('Porcentaje de Anomalías por Video')
                plt.ylabel('Porcentaje (%)')
                plt.xlabel('Video')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig('anomalias_por_video.png')
                print("  ✓ Gráfico de anomalías guardado")
        
        # 5. Distribución de patrones de movimiento
        if 'Patron_Movimiento' in self.combined_data.columns:
            plt.figure(figsize=(10, 6))
            patron_counts = self.combined_data['Patron_Movimiento'].value_counts()
            plt.pie(patron_counts, labels=patron_counts.index, autopct='%1.1f%%',
                   shadow=True, startangle=90)
            plt.axis('equal')
            plt.title('Distribución de Patrones de Movimiento')
            plt.tight_layout()
            plt.savefig('distribucion_patrones.png')
            print("  ✓ Gráfico de patrones guardado")
        
        print("Todas las visualizaciones han sido guardadas en el directorio actual")
    
    def generar_reporte(self, directorio_salida='.'):
        """
        Genera un informe detallado de análisis en formato HTML
        
        Parámetros:
        -----------
        directorio_salida : str
            Directorio donde se guardará el reporte
        """
        if self.combined_data is None:
            print("No hay datos cargados para generar reporte")
            return
        
        # Crear directorio si no existe
        os.makedirs(directorio_salida, exist_ok=True)
        
        print("\nGenerando reporte de análisis...")
        
        # Estructura del reporte HTML
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Análisis de Movimientos en Videos</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 0; color: #333; }}
                .container {{ width: 90%; margin: 0 auto; padding: 20px; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ text-align: left; padding: 12px; }}
                th {{ background-color: #2c3e50; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .graph {{ margin: 30px 0; text-align: center; }}
                .graph img {{ max-width: 100%; height: auto; }}
                .footer {{ margin-top: 50px; text-align: center; color: #7f8c8d; font-size: 0.8em; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Análisis de Movimientos en Videos</h1>
                <p>Fecha de análisis: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}</p>
                
                <h2>Resumen General</h2>
                <table>
                    <tr>
                        <th>Métrica</th>
                        <th>Valor</th>
                    </tr>
                    <tr>
                        <td>Total de Videos Analizados</td>
                        <td>{len(self.dataframes)}</td>
                    </tr>
                    <tr>
                        <td>Total de Registros</td>
                        <td>{len(self.combined_data)}</td>
                    </tr>
                    <tr>
                        <td>Objetos Únicos Detectados</td>
                        <td>{self.combined_data['Objeto'].nunique()}</td>
                    </tr>
                    <tr>
                        <td>Velocidad Media (px/s)</td>
                        <td>{self.combined_data['Velocidad'].mean():.2f}</td>
                    </tr>
                </table>
        """
        
        # Añadir sección de análisis por video
        html_content += """
                <h2>Análisis por Video</h2>
                <table>
                    <tr>
                        <th>Video</th>
                        <th>Frames</th>
                        <th>Objetos</th>
                        <th>Veloc. Media</th>
                        <th>Patrón Dominante</th>
                        <th>% Cíclico</th>
                        <th>% Interacción</th>
                    </tr>
        """
        
        for video_id, df in self.dataframes.items():
            frames = df['Frame'].max() - df['Frame'].min() + 1
            objetos = df['Objeto'].nunique()
            veloc_media = df['Velocidad'].mean()
            
            patron_dominante = "N/A"
            if 'Patron_Movimiento' in df.columns:
                patron_dominante = df['Patron_Movimiento'].value_counts().index[0]
            
            pct_ciclico = df['Es_Ciclico'].mean() * 100 if 'Es_Ciclico' in df.columns else 0
            pct_interaccion = df['En_Interaccion'].mean() * 100 if 'En_Interaccion' in df.columns else 0
            
            html_content += f"""
                    <tr>
                        <td>{video_id}</td>
                        <td>{frames}</td>
                        <td>{objetos}</td>
                        <td>{veloc_media:.2f}</td>
                        <td>{patron_dominante}</td>
                        <td>{pct_ciclico:.1f}%</td>
                        <td>{pct_interaccion:.1f}%</td>
                    </tr>
            """
        
        html_content += """
                </table>
        """
        
        # Añadir sección de anomalías si existen
        if hasattr(self, 'anomalias') and self.anomalias:
            html_content += """
                <h2>Detección de Anomalías</h2>
                <table>
                    <tr>
                        <th>Video</th>
                        <th>Anomalías</th>
                        <th>% del Total</th>
                        <th>Frames Destacados</th>
                    </tr>
            """
            
            for video_id, anomalias_df in self.anomalias.items():
                if len(anomalias_df) > 0:
                    pct = (len(anomalias_df) / len(self.dataframes[video_id])) * 100
                    frames_destacados = anomalias_df['Frame'].value_counts().sort_values(ascending=False).head(3).index.tolist()
                    frames_str = ", ".join(map(str, frames_destacados))
                    
                    html_content += f"""
                        <tr>
                            <td>{video_id}</td>
                            <td>{len(anomalias_df)}</td>
                            <td>{pct:.1f}%</td>
                            <td>{frames_str}</td>
                        </tr>
                    """
            
            html_content += """
                </table>
            """
        
        # Añadir sección de clustering si existe
        if 'cluster' in self.combined_data.columns:
            html_content += """
                <h2>Análisis de Clustering</h2>
                <p>Agrupación de movimientos similares detectados en los videos:</p>
                <table>
                    <tr>
                        <th>Cluster</th>
                        <th>Registros</th>
                        <th>% del Total</th>
                        <th>Características Distintivas</th>
                        <th>Videos Predominantes</th>
                    </tr>
            """
            
            for cluster_id in sorted(self.combined_data['cluster'].unique()):
                cluster_df = self.combined_data[self.combined_data['cluster'] == cluster_id]
                pct = (len(cluster_df) / len(self.combined_data)) * 100
                
                # Características distintivas
                features_numericas = ['Velocidad', 'Linealidad', 'Circularidad', 'Zigzag']
                caracteristicas = []
                
                for feat in features_numericas:
                    cluster_mean = cluster_df[feat].mean()
                    total_mean = self.combined_data[feat].mean()
                    diff = ((cluster_mean - total_mean) / total_mean) * 100 if total_mean != 0 else 0
                    
                    if abs(diff) > 20:  # Solo características significativas
                        direction = "alto" if diff > 0 else "bajo"
                        caracteristicas.append(f"{feat} {direction}")
                
                caracteristicas_str = ", ".join(caracteristicas[:3])  # Top 3
                
                # Videos predominantes
                videos_predominantes = cluster_df['Video'].value_counts(normalize=True).head(2)
                videos_str = ", ".join([f"{v} ({p*100:.1f}%)" for v, p in videos_predominantes.items()])
                
                html_content += f"""
                    <tr>
                        <td>{cluster_id}</td>
                        <td>{len(cluster_df)}</td>
                        <td>{pct:.1f}%</td>
                        <td>{caracteristicas_str}</td>
                        <td>{videos_str}</td>
                    </tr>
                """
            
            html_content += """
                </table>
            """
        
        # Añadir visualizaciones
        html_content += """
                <h2>Visualizaciones</h2>
        """
        
        visualizaciones = [
            ('velocidades_por_video.png', 'Distribución de Velocidades por Video'),
            ('correlaciones_caracteristicas.png', 'Correlaciones entre Características de Movimiento'),
            ('clusters_movimiento.png', 'Clustering de Patrones de Movimiento'),
            ('anomalias_por_video.png', 'Porcentaje de Anomalías por Video'),
            ('distribucion_patrones.png', 'Distribución de Patrones de Movimiento')
        ]
        
        for img_file, titulo in visualizaciones:
            if os.path.exists(img_file):
                html_content += f"""
                <div class="graph">
                    <h3>{titulo}</h3>
                    <img src="{img_file}" alt="{titulo}">
                </div>
                """
        
        # Cerrar documento HTML
        html_content += """
                <div class="footer">
                    <p>Generado automáticamente por MovementAnalyzer</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Guardar reporte
        report_path = os.path.join(directorio_salida, 'reporte_analisis_movimiento.html')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Reporte HTML generado en: {report_path}")

# Función principal para ejecutar el análisis completo
def analizar_datos_movimiento(directorio_csv, directorio_salida='.'):
    """
    Función principal para realizar un análisis completo de los datos
    
    Parámetros:
    -----------
    directorio_csv : str
        Directorio que contiene los archivos CSV a analizar
    directorio_salida : str
        Directorio donde se guardarán los resultados
    """
    print("=" * 60)
    print("ANÁLISIS DE PATRONES DE MOVIMIENTO EN VIDEOS")
    print("=" * 60)
    
    # Inicializar analizador
    analizador = MovementAnalyzer(directorio_csv)
    
    # Cargar datos
    datos_cargados = analizador.cargar_datos()
    if not datos_cargados:
        print("No se pudieron cargar los datos. Terminando análisis.")
        return
    
    # Limpiar datos
    analizador.limpiar_datos()
    
    # Realizar análisis
    analizador.calcular_estadisticas_basicas()
    analizador.detectar_anomalias(contamination=0.05)
    analizador.analisis_clustering(n_clusters=4)
    analizador.visualizar_resultados()
    
    # Generar reporte
    analizador.generar_reporte(directorio_salida)
    
    print("\n" + "=" * 60)
    print("Análisis completado con éxito")
    print("=" * 60)
    print(f"Los resultados han sido guardados en: {directorio_salida}")
    
    return analizador

# Si se ejecuta como script principal
if __name__ == "__main__":
    import argparse
    
    # Configurar argumentos de línea de comandos
    parser = argparse.ArgumentParser(description='Analiza datos de movimiento en videos.')
    parser.add_argument('--input', '-i', required=True, help='Directorio con archivos CSV de movimiento')
    parser.add_argument('--output', '-o', default='.', help='Directorio de salida para resultados')
    parser.add_argument('--clusters', '-c', type=int, default=4, help='Número de clusters para análisis')
    parser.add_argument('--anomalias', '-a', type=float, default=0.05, 
                        help='Proporción esperada de anomalías (0.01-0.1)')
    
    args = parser.parse_args()
    
    # Validar argumentos
    if not os.path.isdir(args.input):
        print(f"Error: El directorio {args.input} no existe")
        sys.exit(1)
    
    # Crear directorio de salida si no existe
    os.makedirs(args.output, exist_ok=True)
    
    # Ejecutar análisis
    try:
        analizar_datos_movimiento(
            directorio_csv=args.input,
            directorio_salida=args.output
        )
    except Exception as e:
        print(f"Error durante el análisis: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)