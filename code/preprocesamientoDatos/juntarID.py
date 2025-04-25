import pandas as pd
import numpy as np
import os
import glob
import math
import cv2
def agregar_como_objeto_unico(df):  
    
    """
    Agrega todos los IDs como si fueran un solo objeto, útil cuando
    sabemos que solo hay una persona en el video.
    """
    df_agregado = df.copy()
    
    # Reemplazar todos los IDs por un único ID
    df_agregado['Objeto'] = 'objeto_principal'
    
    # Agrupar por frame (promediando posiciones si hay más de una detección por frame)
    campos_numericos = df.select_dtypes(include=['float64', 'int64']).columns
    campos_numericos = [c for c in campos_numericos if c not in ['Frame']]
    
    # Agrupar y promediar valores numéricos
    df_agregado = df_agregado.groupby(['Frame', 'Objeto'])[campos_numericos].mean().reset_index()
    
    # Valores categóricos (tomar el más frecuente por frame)
    campos_categoricos = df.select_dtypes(include=['object']).columns
    campos_categoricos = [c for c in campos_categoricos if c not in ['Objeto', 'Video']]
    
    if campos_categoricos:
        # Para cada campo categórico
        for campo in campos_categoricos:
            # Encontrar el valor más frecuente por frame
            valores_frecuentes = df.groupby('Frame')[campo].agg(lambda x: x.value_counts().index[0]).reset_index()
            
            # Unir con el dataframe agregado
            df_agregado = df_agregado.merge(valores_frecuentes, on='Frame', how='left')
    
    # Volver a calcular características que dependen de la trayectoria completa
    # (esto requeriría reimplementar parte de tu extractor de características)
    
    return df_agregado

def fusionar_ids_fragmentados(df):
    """
    Fusiona IDs que probablemente pertenezcan al mismo objeto basado en:
    - Proximidad espacial
    - Proximidad temporal (uno aparece justo cuando otro desaparece)
    - Similitud de características de movimiento
    """
    # Crear copia para no modificar el original
    df_fusion = df.copy()
    
    # Obtener todos los IDs únicos
    todos_ids = df['Objeto'].unique()
    
    # Crear mapping de IDs a fusionar
    mapping_ids = {id_obj: id_obj for id_obj in todos_ids}
    
    # Para cada par de IDs, verificar si deben fusionarse
    for id1 in todos_ids:
        datos_id1 = df[df['Objeto'] == id1]
        
        # Frame mínimo y máximo para este ID
        min_frame_id1 = datos_id1['Frame'].min()
        max_frame_id1 = datos_id1['Frame'].max()
        
        # Última posición conocida
        ultima_pos_id1 = datos_id1[datos_id1['Frame'] == max_frame_id1][['Centroide_X', 'Centroide_Y']].values[0]
        
        # Buscar candidatos para fusión (IDs que aparecen justo después)
        for id2 in todos_ids:
            if id1 == id2 or mapping_ids[id1] == mapping_ids[id2]:
                continue  # Saltar el mismo ID o IDs ya fusionados
                
            datos_id2 = df[df['Objeto'] == id2]
            min_frame_id2 = datos_id2['Frame'].min()
            
            # Verificar proximidad temporal (tolerancia de 5 frames)
            if 0 <= min_frame_id2 - max_frame_id1 <= 5:
                # Primera posición del segundo ID
                primera_pos_id2 = datos_id2[datos_id2['Frame'] == min_frame_id2][['Centroide_X', 'Centroide_Y']].values[0]
                
                # Calcular distancia espacial
                distancia = np.sqrt(np.sum((ultima_pos_id1 - primera_pos_id2)**2))
                
                # Si la distancia es pequeña, fusionar IDs
                if distancia < 50:  # Umbral ajustable
                    # Asignar al ID mayor el valor del ID menor (por convención)
                    id_menor = min(mapping_ids[id1], mapping_ids[id2])
                    id_mayor = max(mapping_ids[id1], mapping_ids[id2])
                    
                    # Actualizar mapping para id_mayor y todos los que apuntan a él
                    for id_tmp in todos_ids:
                        if mapping_ids[id_tmp] == id_mayor:
                            mapping_ids[id_tmp] = id_menor
    
    # Aplicar la fusión de IDs
    for id_orig, id_nuevo in mapping_ids.items():
        if id_orig != id_nuevo:
            df_fusion.loc[df_fusion['Objeto'] == id_orig, 'Objeto'] = id_nuevo
    
    return df_fusion

def recalcular_caracteristicas_trayectoria(df):
    """
    Recalcula las características basadas en trayectoria después de fusionar o agregar IDs.
    Reimplementa los cálculos de características del FeatureExtractor original.
    
    Parámetros:
    -----------
    df : DataFrame
        DataFrame con los datos después de fusionar o agregar IDs
        
    Retorna:
    --------
    DataFrame con las características recalculadas
    """
    # Crear copia para no modificar el original
    df_recalc = df.copy()
    
    # Procesar por objeto
    objetos_unicos = df_recalc['Objeto'].unique()
    
    # Preparar para almacenar características calculadas
    caracteristicas_calculadas = []
    
    # Para cada objeto, recalcular características basadas en trayectoria
    for obj_id in objetos_unicos:
        # Obtener datos de este objeto ordenados por frame
        datos_obj = df_recalc[df_recalc['Objeto'] == obj_id].sort_values('Frame')
        
        # Si hay muy pocos frames, no podemos calcular trayectoria adecuadamente
        if len(datos_obj) < 5:
            # Añadir datos tal como están
            caracteristicas_calculadas.append(datos_obj)
            continue
        
        # Obtener trayectoria (centroides)
        centroides = []
        for _, row in datos_obj.iterrows():
            if 'Centroide_X' in row and 'Centroide_Y' in row:
                centroides.append((row['Centroide_X'], row['Centroide_Y']))
            else:
                # Si no hay centroides, no podemos calcular trayectoria
                break
        
        # Si no tenemos suficientes centroides, continuar con el siguiente objeto
        if len(centroides) < 5:
            caracteristicas_calculadas.append(datos_obj)
            continue
        
        # Convertir a numpy array para cálculos vectorizados
        tray = np.array(centroides)
        
        # 1. Calcular métricas básicas (desplazamiento, velocidad, aceleración, dirección)
        datos_obj_new = datos_obj.copy()
        for i in range(1, len(datos_obj)):
            # Obtener frames actual y anterior
            idx_curr = datos_obj.index[i]
            idx_prev = datos_obj.index[i-1]
            
            # Calcular desplazamiento
            dx = tray[i, 0] - tray[i-1, 0]
            dy = tray[i, 1] - tray[i-1, 1]
            desplazamiento = np.sqrt(dx**2 + dy**2)
            
            # Calcular velocidad (si tenemos FPS)
            fps = 25  # Valor por defecto
            velocidad = desplazamiento * fps
            
            # Calcular dirección (ángulo)
            angulo = math.degrees(math.atan2(dy, dx))
            
            # Actualizar datos
            datos_obj_new.loc[idx_curr, 'Desplazamiento'] = desplazamiento
            datos_obj_new.loc[idx_curr, 'Velocidad'] = velocidad
            datos_obj_new.loc[idx_curr, 'Direccion'] = angulo
            
            # Calcular aceleración si tenemos frames suficientes
            if i > 1:
                idx_prev2 = datos_obj.index[i-2]
                v_prev = datos_obj_new.loc[idx_prev, 'Velocidad']
                datos_obj_new.loc[idx_curr, 'Aceleracion'] = (velocidad - v_prev) * fps
            
        # 2. Analizar trayectoria completa (linealidad, circularidad, zigzag)
        
        # Calcular la distancia total recorrida
        dist_total = np.sum(np.sqrt(np.sum(np.diff(tray, axis=0)**2, axis=1)))
        
        # Calcular la distancia directa desde el inicio al fin
        dist_directa = np.sqrt(np.sum((tray[-1] - tray[0])**2))
        
        # Calcular linealidad (1 = perfectamente lineal)
        linealidad = dist_directa / (dist_total + 1e-6) if dist_total > 0 else 0
        
        # Calcular cambios bruscos de dirección (zigzag)
        if len(tray) > 2:
            vectores = np.diff(tray, axis=0)
            # Normalizar vectores
            normas = np.linalg.norm(vectores, axis=1)
            normas = np.where(normas == 0, 1e-10, normas)  # Evitar división por cero
            vectores_norm = vectores / normas[:, np.newaxis]
            
            # Calcular ángulos entre vectores consecutivos
            productos = np.sum(vectores_norm[:-1] * vectores_norm[1:], axis=1)
            productos = np.clip(productos, -1, 1)  # Evitar errores numéricos
            angulos = np.arccos(productos)
            
            # Calcular zigzag como promedio de cambios de dirección
            zigzag = np.mean(np.abs(angulos)) / np.pi if len(angulos) > 0 else 0
        else:
            zigzag = 0
        
        # Calcular circularidad
        if len(tray) > 3:
            # Centro aproximado de la trayectoria
            centro = np.mean(tray, axis=0)
            # Distancias al centro
            distancias = np.sqrt(np.sum((tray - centro)**2, axis=1))
            # Variación de las distancias (menor = más circular)
            circularidad = 1 - (np.std(distancias) / (np.mean(distancias) + 1e-6))
        else:
            circularidad = 0
            
        # Determinar patrón predominante
        if linealidad > 0.8:
            patron = "Lineal"
        elif circularidad > 0.7:
            patron = "Circular"
        elif zigzag > 0.5:
            patron = "Zigzag"
        else:
            patron = "Mixto"
        
        # 3. Detectar comportamiento cíclico
        ventana_tiempo = min(20, len(tray))
        
        # Analizar ciclos en coordenadas x e y
        x = tray[:, 0]
        y = tray[:, 1]
        
        # Calcular autocorrelación para detectar patrones repetitivos
        autocorr_x = np.correlate(x - np.mean(x), x - np.mean(x), mode='full')
        autocorr_y = np.correlate(y - np.mean(y), y - np.mean(y), mode='full')
        
        # Normalizar
        autocorr_x = autocorr_x[len(autocorr_x)//2:] / (autocorr_x[len(autocorr_x)//2] + 1e-10)
        autocorr_y = autocorr_y[len(autocorr_y)//2:] / (autocorr_y[len(autocorr_y)//2] + 1e-10)
        
        # Buscar picos en la autocorrelación (indican periodicidad)
        if len(autocorr_x) > 2:
            picos_x = (autocorr_x[1:-1] > autocorr_x[:-2]) & (autocorr_x[1:-1] > autocorr_x[2:])
            picos_y = (autocorr_y[1:-1] > autocorr_y[:-2]) & (autocorr_y[1:-1] > autocorr_y[2:])
            
            hay_ciclo_x = np.any(picos_x) and np.max(autocorr_x[1:]) > 0.5
            hay_ciclo_y = np.any(picos_y) and np.max(autocorr_y[1:]) > 0.5
            
            es_ciclico = hay_ciclo_x or hay_ciclo_y
            
            # Calcular frecuencia y amplitud si es cíclico
            if es_ciclico:
                if np.max(autocorr_x[1:]) > np.max(autocorr_y[1:]):
                    indices_picos = np.where(picos_x)[0] + 1  # +1 por el slice
                    amplitud = np.std(x)
                else:
                    indices_picos = np.where(picos_y)[0] + 1
                    amplitud = np.std(y)
                    
                if len(indices_picos) > 0:
                    periodo = indices_picos[0]  # Frames entre repeticiones
                    frecuencia = 1 / periodo if periodo > 0 else 0
                else:
                    frecuencia = 0
                    amplitud = 0
            else:
                frecuencia = 0
                amplitud = 0
        else:
            es_ciclico = False
            frecuencia = 0
            amplitud = 0
        
        # 4. Calcular área del casco convexo (dispersión)
        if len(tray) > 2:
            try:
                hull = cv2.convexHull(tray.astype(np.float32))
                area_hull = cv2.contourArea(hull)
            except:
                area_hull = 0
        else:
            area_hull = 0
        
        # 5. Asignar características calculadas a todos los registros de este objeto
        datos_obj_new['Linealidad'] = linealidad
        datos_obj_new['Circularidad'] = circularidad
        datos_obj_new['Zigzag'] = zigzag
        datos_obj_new['Patron_Movimiento'] = patron
        datos_obj_new['Es_Ciclico'] = 1 if es_ciclico else 0
        datos_obj_new['Frecuencia_Ciclo'] = frecuencia
        datos_obj_new['Amplitud_Ciclo'] = amplitud
        datos_obj_new['Area_Trayectoria'] = area_hull
        
        # 6. Agregar a los resultados
        caracteristicas_calculadas.append(datos_obj_new)
    
    # Combinar todos los resultados
    df_final = pd.concat(caracteristicas_calculadas, ignore_index=True)
    
    # 7. Actualizar interacciones entre objetos (solo si hay múltiples objetos)
    if len(objetos_unicos) > 1:
        # Inicializar columna de interacción
        df_final['En_Interaccion'] = 0
        
        # Procesar por frame
        for frame in df_final['Frame'].unique():
            datos_frame = df_final[df_final['Frame'] == frame]
            
            # Si hay al menos 2 objetos en este frame
            if len(datos_frame) > 1:
                centroides = {}
                for _, row in datos_frame.iterrows():
                    if 'Centroide_X' in row and 'Centroide_Y' in row:
                        centroides[row['Objeto']] = (row['Centroide_X'], row['Centroide_Y'])
                
                # Verificar distancias entre todos los pares
                for obj1 in centroides:
                    for obj2 in centroides:
                        if obj1 != obj2:
                            # Calcular distancia
                            c1 = np.array(centroides[obj1])
                            c2 = np.array(centroides[obj2])
                            distancia = np.sqrt(np.sum((c1 - c2)**2))
                            
                            # Si están cerca, marcar como interacción
                            if distancia < 50:  # Umbral ajustable
                                df_final.loc[
                                    (df_final['Frame'] == frame) & 
                                    ((df_final['Objeto'] == obj1) | (df_final['Objeto'] == obj2)),
                                    'En_Interaccion'
                                ] = 1
    
    return df_final


def procesamiento_inteligente_ids(df, max_objetos_reales=None):
    """
    Procesa IDs fragmentados identificando el número real de objetos en la escena
    mediante análisis de superposición temporal y agrupamiento espacial.
    
    Parámetros:
    -----------
    df : DataFrame
        DataFrame con los datos de trayectorias
    max_objetos_reales : int o None
        Número máximo de objetos reales esperados en la escena.
        Si es None, se estima automáticamente.
        
    Retorna:
    --------
    DataFrame con IDs corregidos
    """
    from sklearn.cluster import DBSCAN
    import numpy as np
    
    # Crear copia para no modificar el original
    df_procesado = df.copy()
    
    # 1. Análisis de simultaneidad para estimar objetos reales
    # Contar cuántos objetos aparecen simultáneamente en cada frame
    objetos_por_frame = df.groupby('Frame')['Objeto'].nunique()
    max_simultaneos = objetos_por_frame.max()
    
    # Si hay un valor proporcionado, usar ese
    if max_objetos_reales is None:
        # Estimar el número real de objetos basado en la simultaneidad
        # y un análisis de la distribución
        conteo_simultaneidad = objetos_por_frame.value_counts().sort_index(ascending=False)
        
        # Encontrar el valor más frecuente entre los conteos altos
        # (ignorando frames con pocos objetos que pueden ser fallos de detección)
        percentil_90 = np.percentile(objetos_por_frame, 90)
        conteos_significativos = conteo_simultaneidad[conteo_simultaneidad.index >= max(2, percentil_90 * 0.8)]
        
        if not conteos_significativos.empty:
            # Usar el conteo simultáneo más frecuente como estimación
            max_objetos_reales = conteos_significativos.index[0]
        else:
            # Fallback a un valor basado en percentil
            max_objetos_reales = max(1, int(percentil_90))
    
    print(f"Número estimado de objetos reales en la escena: {max_objetos_reales}")
    
    # 2. Agrupar trayectorias por proximidad espacial y temporal
    # Para cada frame, extraer centroides
    centroides_por_frame = {}
    objetos_por_centroide = {}
    
    for frame in df['Frame'].unique():
        datos_frame = df[df['Frame'] == frame]
        
        centroides = []
        objetos = []
        
        for _, row in datos_frame.iterrows():
            if 'Centroide_X' in row and 'Centroide_Y' in row:
                centroides.append([row['Centroide_X'], row['Centroide_Y']])
                objetos.append(row['Objeto'])
        
        if centroides:
            centroides_por_frame[frame] = np.array(centroides)
            objetos_por_centroide[frame] = objetos
    
    # 3. Asignar IDs consistentes basados en agrupamiento espacial
    nuevo_mapping = {}  # Mapeo de objeto original a nuevo ID
    contador_nuevos_ids = 0
    
    # Procesar frames en orden
    for frame in sorted(centroides_por_frame.keys()):
        if frame not in centroides_por_frame or not len(centroides_por_frame[frame]):
            continue
            
        centroides = centroides_por_frame[frame]
        objetos = objetos_por_centroide[frame]
        
        # Aplicar DBSCAN para agrupar centroides cercanos
        if len(centroides) > 1:
            # Estimar epsilon basado en la distribución de distancias
            todas_distancias = []
            for i in range(len(centroides)):
                for j in range(i+1, len(centroides)):
                    dist = np.linalg.norm(centroides[i] - centroides[j])
                    todas_distancias.append(dist)
            
            # Usar un percentil bajo como epsilon para ser conservador
            if todas_distancias:
                epsilon = np.percentile(todas_distancias, 25)  # Ajustable
                # Asegurar que epsilon es mayor que 0
                epsilon = max(epsilon, 0.1)  # Usamos 0.1 como mínimo para evitar errores
            else:
                epsilon = 50  # Valor por defecto
            
            # Agrupar centroides
            clustering = DBSCAN(eps=epsilon, min_samples=1).fit(centroides)
            labels = clustering.labels_
        else:
            # Solo hay un objeto
            labels = np.array([0])
        
        # Número de clusters en este frame
        n_clusters = len(set(labels))
        
        # Si hay más clusters que objetos reales esperados, fusionar los más cercanos
        if n_clusters > max_objetos_reales:
            # Calcular distancias entre centroides de clusters
            centros_clusters = []
            for i in range(n_clusters):
                if np.sum(labels == i) > 0:
                    centros_clusters.append(np.mean(centroides[labels == i], axis=0))
            
            # Calcular matriz de distancias entre clusters
            dist_matrix = np.zeros((n_clusters, n_clusters))
            for i in range(n_clusters):
                for j in range(i+1, n_clusters):
                    dist_matrix[i, j] = np.linalg.norm(centros_clusters[i] - centros_clusters[j])
                    dist_matrix[j, i] = dist_matrix[i, j]
            
            # Fusionar clusters cercanos hasta tener max_objetos_reales
            while n_clusters > max_objetos_reales:
                # Encontrar par más cercano (ignorando diagonales de ceros)
                mask = dist_matrix > 0
                if not np.any(mask):
                    break  # No hay más pares para fusionar
                
                i, j = np.unravel_index(np.where(mask, dist_matrix, np.inf).argmin(), dist_matrix.shape)
                
                # Fusionar clusters i y j (todos los j pasan a ser i)
                labels[labels == j] = i
                
                # Actualizar matriz de distancias (eliminar j)
                dist_matrix[i, :] = 0
                dist_matrix[:, i] = 0
                dist_matrix[j, :] = 0
                dist_matrix[:, j] = 0
                
                # Recalcular clusters únicos
                n_clusters = len(set([l for l in labels if l >= 0]))
        
        # Para cada cluster, asignar o reutilizar IDs
        for cluster_idx in set(labels):
            # Obtener objetos originales en este cluster
            indices_cluster = np.where(labels == cluster_idx)[0]
            objetos_cluster = [objetos[i] for i in indices_cluster]
            
            # Verificar si alguno de estos objetos ya tiene asignado un nuevo ID
            nuevos_ids_existentes = [nuevo_mapping.get(obj) for obj in objetos_cluster 
                                    if obj in nuevo_mapping]
            
            if nuevos_ids_existentes:
                # Usar el ID más frecuente
                from collections import Counter
                id_counter = Counter([id for id in nuevos_ids_existentes if id is not None])
                if id_counter:
                    nuevo_id = id_counter.most_common(1)[0][0]
                else:
                    # Crear nuevo ID si todos son None
                    nuevo_id = f"objeto_real_{contador_nuevos_ids}"
                    contador_nuevos_ids += 1
            else:
                # Crear nuevo ID
                nuevo_id = f"objeto_real_{contador_nuevos_ids}"
                contador_nuevos_ids += 1
            
            # Asignar nuevo ID a todos los objetos en el cluster
            for obj in objetos_cluster:
                nuevo_mapping[obj] = nuevo_id
    
    # 4. Aplicar el nuevo mapping al DataFrame
    # Para objetos no procesados (en frames sin centroides), mantener ID original
    for obj in df['Objeto'].unique():
        if obj not in nuevo_mapping:
            nuevo_mapping[obj] = obj
    
    # Aplicar mapping
    df_procesado['Objeto_Original'] = df_procesado['Objeto']  # Guardar ID original
    df_procesado['Objeto'] = df_procesado['Objeto'].map(nuevo_mapping)
    
    # Verificar resultado
    nuevos_objetos = df_procesado['Objeto'].nunique()
    print(f"IDs originales: {df['Objeto'].nunique()}, IDs después de procesamiento: {nuevos_objetos}")
    
    return df_procesado

def procesar_dataset_con_ids_fragmentados(directorio_csv, directorio_salida):
    """
    Procesa un dataset completo con IDs fragmentados
    """
    # Crear directorio de salida
    os.makedirs(directorio_salida, exist_ok=True)
    
    # Procesar cada archivo
    for csv_file in glob.glob(os.path.join(directorio_csv, "*.csv")):
        nombre = os.path.basename(csv_file).split('.')[0]
        print(f"Procesando {nombre}...")
        
        # Cargar datos
        df = pd.read_csv(csv_file)
        
        # Procesar IDs fragmentados - enfoque unificado para todos los casos
        df_procesado = procesamiento_inteligente_ids(df)
        
        # Recalcular características basadas en trayectoria
        df_procesado = recalcular_caracteristicas_trayectoria(df_procesado)
        
        # Guardar resultado
        df_procesado.to_csv(os.path.join(directorio_salida, f"{nombre}_procesado.csv"), index=False)
        
directorio_csv = r'D:\Documentos\Monitoreo-Comportamientos-Sospechosos\datasetCSV'
directorio_salida = r'D:\Documentos\Monitoreo-Comportamientos-Sospechosos\datasetCSV_procesado'

procesar_dataset_con_ids_fragmentados(directorio_csv, directorio_salida)