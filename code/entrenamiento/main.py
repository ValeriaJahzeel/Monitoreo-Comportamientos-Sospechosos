"""
Script principal para el sistema de clasificación de comportamientos sospechosos.
Este script coordina todo el proceso: procesamiento de datos, ingeniería de características,
entrenamiento de modelos y evaluación de rendimiento.
"""
import os
import argparse
import time
from datetime import datetime

# Importar módulos del sistema
from data_processing import DataProcessor
from feature_engineering import FeatureEngineer
from model_training import ModelTrainer
from model_evaluation import ModelEvaluator

def setup_directories(base_dir='.'):
    """
    Configura la estructura de directorios para el proyecto.
    
    Parámetros:
    -----------
    base_dir : str
        Directorio base del proyecto
    """
    # Crear directorios principales
    dirs = [
        os.path.join(base_dir, 'processed_data'),
        os.path.join(base_dir, 'processed_data', 'feature_analysis'),
        os.path.join(base_dir, 'models'),
        os.path.join(base_dir, 'models', 'mlp'),
        os.path.join(base_dir, 'models', 'rf'),
        os.path.join(base_dir, 'models', 'lstm'),
        os.path.join(base_dir, 'evaluation'),
        os.path.join(base_dir, 'evaluation', 'plots'),
        os.path.join(base_dir, 'evaluation', 'reports'),
        os.path.join(base_dir, 'logs')
    ]
    
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
    
    return dirs

def process_data(data_dir, output_dir, log_file):
    """
    Ejecuta el procesamiento de datos.
    
    Parámetros:
    -----------
    data_dir : str
        Directorio con los datos CSV originales
    output_dir : str
        Directorio donde se guardarán los datos procesados
    log_file : str
        Archivo para guardar el registro de ejecución
    """
    with open(log_file, 'a') as log:
        log.write(f"\n{'='*50}\n")
        log.write(f"PROCESAMIENTO DE DATOS\n")
        log.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write(f"{'='*50}\n")
    
    print(f"\n{'='*50}")
    print(f"PROCESAMIENTO DE DATOS")
    print(f"{'='*50}")
    
    try:
        # Crear y ejecutar procesador de datos
        processor = DataProcessor(data_dir=data_dir, output_dir=output_dir)
        results = processor.process_all()
        
        with open(log_file, 'a') as log:
            log.write("Procesamiento de datos completado con éxito.\n")
        
        print("Procesamiento de datos completado con éxito.")
        return True
    except Exception as e:
        with open(log_file, 'a') as log:
            log.write(f"Error en procesamiento de datos: {str(e)}\n")
        
        print(f"Error en procesamiento de datos: {str(e)}")
        return False

def analyze_features(data_dir, log_file):
    """
    Ejecuta el análisis de características.
    
    Parámetros:
    -----------
    data_dir : str
        Directorio con los datos procesados
    log_file : str
        Archivo para guardar el registro de ejecución
    """
    with open(log_file, 'a') as log:
        log.write(f"\n{'='*50}\n")
        log.write(f"ANÁLISIS DE CARACTERÍSTICAS\n")
        log.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write(f"{'='*50}\n")
    
    print(f"\n{'='*50}")
    print(f"ANÁLISIS DE CARACTERÍSTICAS")
    print(f"{'='*50}")
    
    try:
        # Crear y ejecutar ingeniero de características
        engineer = FeatureEngineer(data_dir=data_dir)
        results = engineer.analyze_all_levels()
        
        with open(log_file, 'a') as log:
            log.write("Análisis de características completado con éxito.\n")
        
        print("Análisis de características completado con éxito.")
        return True
    except Exception as e:
        with open(log_file, 'a') as log:
            log.write(f"Error en análisis de características: {str(e)}\n")
        
        print(f"Error en análisis de características: {str(e)}")
        return False

def train_models(data_dir, models_dir, log_file):
    """
    Ejecuta el entrenamiento de modelos.
    
    Parámetros:
    -----------
    data_dir : str
        Directorio con los datos procesados
    models_dir : str
        Directorio donde se guardarán los modelos
    log_file : str
        Archivo para guardar el registro de ejecución
    """
    with open(log_file, 'a') as log:
        log.write(f"\n{'='*50}\n")
        log.write(f"ENTRENAMIENTO DE MODELOS\n")
        log.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write(f"{'='*50}\n")
    
    print(f"\n{'='*50}")
    print(f"ENTRENAMIENTO DE MODELOS")
    print(f"{'='*50}")
    
    try:
        # Crear y ejecutar entrenador de modelos
        trainer = ModelTrainer(data_dir=data_dir, output_dir=models_dir)
        results = trainer.train_all_models()
        
        with open(log_file, 'a') as log:
            log.write("Entrenamiento de modelos completado con éxito.\n")
        
        print("Entrenamiento de modelos completado con éxito.")
        return True
    except Exception as e:
        with open(log_file, 'a') as log:
            log.write(f"Error en entrenamiento de modelos: {str(e)}\n")
        
        print(f"Error en entrenamiento de modelos: {str(e)}")
        return False

def evaluate_models(models_dir, data_dir, eval_dir, log_file):
    """
    Ejecuta la evaluación de modelos.
    
    Parámetros:
    -----------
    models_dir : str
        Directorio con los modelos entrenados
    data_dir : str
        Directorio con los datos procesados
    eval_dir : str
        Directorio donde se guardarán los resultados de evaluación
    log_file : str
        Archivo para guardar el registro de ejecución
    """
    with open(log_file, 'a') as log:
        log.write(f"\n{'='*50}\n")
        log.write(f"EVALUACIÓN DE MODELOS\n")
        log.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write(f"{'='*50}\n")
    
    print(f"\n{'='*50}")
    print(f"EVALUACIÓN DE MODELOS")
    print(f"{'='*50}")
    
    try:
        # Crear y ejecutar evaluador de modelos
        evaluator = ModelEvaluator(models_dir=models_dir, data_dir=data_dir, output_dir=eval_dir)
        results = evaluator.generate_full_report()
        
        with open(log_file, 'a') as log:
            log.write("Evaluación de modelos completada con éxito.\n")
            log.write("\nResumen de rendimiento:\n")
            for _, row in results.iterrows():
                log.write(f"  {row['model_name']}: ")
                log.write(f"Exactitud={row['accuracy']:.4f}, ")
                log.write(f"F1-Score={row['f1']:.4f}\n")
        
        print("Evaluación de modelos completada con éxito.")
        return True
    except Exception as e:
        with open(log_file, 'a') as log:
            log.write(f"Error en evaluación de modelos: {str(e)}\n")
        
        print(f"Error en evaluación de modelos: {str(e)}")
        return False

def main():
    """Función principal que coordina todo el proceso."""
    # Configurar parser de argumentos
    parser = argparse.ArgumentParser(description='Sistema de clasificación de comportamientos sospechosos')
    
    parser.add_argument('--data_dir', default='./informacion/csv/',
                       help='Directorio con los datos CSV originales')
    parser.add_argument('--output_dir', default='./results/',
                       help='Directorio para almacenar resultados')
    parser.add_argument('--skip_processing', action='store_true',
                       help='Omitir etapa de procesamiento de datos')
    parser.add_argument('--skip_features', action='store_true',
                       help='Omitir etapa de análisis de características')
    parser.add_argument('--skip_training', action='store_true',
                       help='Omitir etapa de entrenamiento de modelos')
    parser.add_argument('--skip_evaluation', action='store_true',
                       help='Omitir etapa de evaluación de modelos')
    
    args = parser.parse_args()
    
    # Configurar directorios
    base_dir = args.output_dir
    setup_directories(base_dir)
    
    # Configurar rutas
    data_dir = args.data_dir
    processed_dir = os.path.join(base_dir, 'processed_data')
    models_dir = os.path.join(base_dir, 'models')
    eval_dir = os.path.join(base_dir, 'evaluation')
    log_dir = os.path.join(base_dir, 'logs')
    
    # Crear archivo de log
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'execution_{timestamp}.log')
    
    with open(log_file, 'w') as log:
        log.write(f"SISTEMA DE CLASIFICACIÓN DE COMPORTAMIENTOS SOSPECHOSOS\n")
        log.write(f"Fecha de ejecución: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write(f"Directorio de datos: {data_dir}\n")
        log.write(f"Directorio de resultados: {base_dir}\n\n")
    
    # Medir tiempo total
    start_time = time.time()
    
    # Ejecutar etapas según configuración
    if not args.skip_processing:
        process_data(data_dir, processed_dir, log_file)
    else:
        print("Omitiendo etapa de procesamiento de datos...")
    
    if not args.skip_features:
        analyze_features(processed_dir, log_file)
    else:
        print("Omitiendo etapa de análisis de características...")
    
    if not args.skip_training:
        train_models(processed_dir, models_dir, log_file)
    else:
        print("Omitiendo etapa de entrenamiento de modelos...")
    
    if not args.skip_evaluation:
        evaluate_models(models_dir, processed_dir, eval_dir, log_file)
    else:
        print("Omitiendo etapa de evaluación de modelos...")
    
    # Calcular tiempo total
    total_time = time.time() - start_time
    
    with open(log_file, 'a') as log:
        log.write(f"\n{'='*50}\n")
        log.write(f"FINALIZACIÓN DEL PROCESO\n")
        log.write(f"Tiempo total de ejecución: {total_time:.2f} segundos\n")
        log.write(f"{'='*50}\n")
    
    print(f"\n{'='*50}")
    print(f"PROCESO COMPLETADO")
    print(f"Tiempo total de ejecución: {total_time:.2f} segundos")
    print(f"Resultados guardados en: {base_dir}")
    print(f"Registro de ejecución: {log_file}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()