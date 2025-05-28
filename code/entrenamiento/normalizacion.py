import pandas as pd
import os

def convertir_caracteristicas_textuales(archivo_entrada, archivo_salida=None):
    """
    Convierte características textuales a numéricas en un archivo CSV.
    
    Args:
        archivo_entrada (str): Ruta al archivo CSV de entrada
        archivo_salida (str, optional): Ruta para guardar el resultado. Si es None, se sobreescribe el original.
    """
    
    # Mapeos para conversión
    mapeo_postura = {
        'Vertical': 0,
        'Horizontal': 1,
        'Neutral': 2
    }
    
    mapeo_patron = {
        'Zigzag': 0,
        'Indeterminado': 1,
        'Circular': 2,
        'Lineal': 3,
        'Mixto': 4
    }
    
    # Leer el archivo CSV
    df = pd.read_csv(archivo_entrada)
    
    # Convertir columnas textuales
    if 'Postura' in df.columns:
        df['Postura'] = df['Postura'].map(mapeo_postura).fillna(-1).astype(int)
    
    if 'Patron_Movimiento' in df.columns:
        df['Patron_Movimiento'] = df['Patron_Movimiento'].map(mapeo_patron).fillna(-1).astype(int)
    
    # Guardar el resultado
    if archivo_salida is None:
        archivo_salida = archivo_entrada
    
    df.to_csv(archivo_salida, index=False)
    print(f"Archivo procesado y guardado en: {archivo_salida}")

def procesar_directorio(directorio, extension='.csv'):
    """
    Procesa todos los archivos con una extensión específica en un directorio.
    
    Args:
        directorio (str): Ruta al directorio con los archivos
        extension (str): Extensión de los archivos a procesar
    """
    for archivo in os.listdir(directorio):
        if archivo.endswith(extension):
            ruta_completa = os.path.join(directorio, archivo)
            print(f"Procesando: {ruta_completa}")
            convertir_caracteristicas_textuales(ruta_completa)

# Ejemplo de uso para un solo archivo:
# convertir_caracteristicas_textuales('merodeo_8_procesado.csv', 'merodeo_8_procesado_numerico.csv')

# Ejemplo de uso para un directorio completo:
procesar_directorio(r'D:\Documentos\Monitoreo-Comportamientos-Sospechosos\csv\normal')