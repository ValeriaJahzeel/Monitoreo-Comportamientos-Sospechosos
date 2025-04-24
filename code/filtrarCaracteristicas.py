import pandas as pd
import os
import glob

def extraer_caracteristicas_relevantes(ruta_entrada, ruta_salida):
    """
    Extrae las características más relevantes de un archivo CSV basado
    en el análisis de correlación de las matrices de movimiento.
    
    Args:
        ruta_entrada: Ruta al archivo CSV de entrada
        ruta_salida: Ruta donde guardar el archivo reducido
    """
    # Características relevantes identificadas en las matrices de correlación
    caract_relevantes = [
        # Columnas de identificación y básicas
        'Frame', 'Objeto', 'Centroide_X', 'Centroide_Y',
        # Características con correlaciones significativas
        'Velocidad',  # En lugar de Desplazamiento (correlación perfecta)
        'Aceleracion',
        'Linealidad',
        'Circularidad',
        'Zigzag',
        'Area_Trayectoria',
        # Características categóricas importantes
        'Patron_Movimiento',
        'Postura',
        # Características binarias relevantes
        'Es_Ciclico'
    ]
    
    try:
        # Cargar el archivo CSV
        df = pd.read_csv(ruta_entrada)
        print(f"Archivo cargado: {ruta_entrada}")
        print(f"Dimensiones originales: {df.shape}")
        
        # Verificar que todas las columnas existen en el DataFrame
        columnas_existentes = [col for col in caract_relevantes if col in df.columns]
        columnas_faltantes = [col for col in caract_relevantes if col not in df.columns]
        
        if columnas_faltantes:
            print(f"Advertencia: Las siguientes columnas no están en el archivo: {columnas_faltantes}")
        
        # Seleccionar solo las columnas relevantes que existen
        df_relevante = df[columnas_existentes]
        
        print(f"Dimensiones del archivo reducido: {df_relevante.shape}")
        
        # Guardar el archivo
        df_relevante.to_csv(ruta_salida, index=False)
        print(f"Archivo guardado con éxito en: {ruta_salida}")
        
        return True
    
    except Exception as e:
        print(f"Error al procesar el archivo {ruta_entrada}: {str(e)}")
        return False

def procesar_directorio(dir_entrada, dir_salida):
    """
    Procesa todos los archivos CSV en un directorio.
    
    Args:
        dir_entrada: Directorio donde están los archivos CSV originales
        dir_salida: Directorio donde guardar los archivos reducidos
    """
    # Crear directorio de salida si no existe
    os.makedirs(dir_salida, exist_ok=True)
    
    # Obtener todos los archivos CSV en el directorio
    archivos_csv = glob.glob(os.path.join(dir_entrada, "*.csv"))
    
    if not archivos_csv:
        print(f"No se encontraron archivos CSV en {dir_entrada}")
        return
    
    print(f"Se encontraron {len(archivos_csv)} archivos CSV en {dir_entrada}")
    
    for archivo in archivos_csv:
        nombre_archivo = os.path.basename(archivo)
        ruta_salida = os.path.join(dir_salida, nombre_archivo.replace(".csv", "_reducido.csv"))
        
        print(f"\nProcesando: {nombre_archivo}")
        extraer_caracteristicas_relevantes(archivo, ruta_salida)
    
    print("\nProcesamiento completado.")

# Ejemplo de uso
if __name__ == "__main__":
    dir_entrada = "D:\Documentos\Monitoreo-Comportamientos-Sospechosos\datasetCSV"  # Reemplaza con la ruta a tus archivos CSV
    dir_salida = r'D:\Documentos\Monitoreo-Comportamientos-Sospechosos\nuevoDatasetCSV'  # Reemplaza con la ruta donde quieres guardar los archivos reducidos
    
    # También puedes procesar un solo archivo
    # archivo_entrada = "ruta/a/tu/archivo.csv"
    # archivo_salida = "ruta/a/tu/archivo_reducido.csv"
    # extraer_caracteristicas_relevantes(archivo_entrada, archivo_salida)
    
    # O procesar todo un directorio
    procesar_directorio(dir_entrada, dir_salida)