"""
Módulo para evaluación de modelos de clasificación de comportamientos.
Este script evalúa los modelos entrenados y genera visualizaciones de rendimiento.
"""
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
import tensorflow as tf
from tensorflow.keras.models import load_model

class ModelEvaluator:
    def __init__(self, models_dir='./models/', data_dir='./processed_data/', output_dir='./evaluation/'):
        """
        Inicializa el evaluador de modelos.
        
        Parámetros:
        -----------
        models_dir : str
            Directorio donde se encuentran los modelos entrenados
        data_dir : str
            Directorio donde se encuentran los datos procesados
        output_dir : str
            Directorio donde se guardarán los resultados de evaluación
        """
        self.models_dir = models_dir
        self.data_dir = data_dir
        self.output_dir = output_dir
        
        # Crear directorios de salida
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'reports'), exist_ok=True)
        
        # Diccionario para almacenar modelos cargados
        self.models = {}
        
        # Diccionario para almacenar resultados
        self.results = {}
        
        # Etiquetas de clase
        self.class_names = ['normal', 'merodeo', 'forcejeo']
    
    def load_models(self):
        """
        Carga todos los modelos entrenados.
        """
        print("Cargando modelos entrenados...")
        
        # Cargar modelos MLP y RF para cada nivel
        for level in ['video', 'object', 'frame']:
            # Cargar modelo MLP
            mlp_path = os.path.join(self.models_dir, 'mlp', f'{level}_mlp.pkl')
            if os.path.exists(mlp_path):
                with open(mlp_path, 'rb') as f:
                    self.models[f'mlp_{level}'] = pickle.load(f)
                print(f"  ✓ Cargado modelo MLP para nivel '{level}'")
            else:
                print(f"  ✗ No se encontró modelo MLP para nivel '{level}'")
            
            # Cargar modelo Random Forest
            rf_path = os.path.join(self.models_dir, 'rf', f'{level}_rf.pkl')
            if os.path.exists(rf_path):
                with open(rf_path, 'rb') as f:
                    self.models[f'rf_{level}'] = pickle.load(f)
                print(f"  ✓ Cargado modelo Random Forest para nivel '{level}'")
            else:
                print(f"  ✗ No se encontró modelo Random Forest para nivel '{level}'")
        
        # Cargar modelo LSTM
        lstm_path = os.path.join(self.models_dir, 'lstm', 'final_model.h5')
        if os.path.exists(lstm_path):
            self.models['lstm'] = load_model(lstm_path)
            print(f"  ✓ Cargado modelo LSTM")
        else:
            print(f"  ✗ No se encontró modelo LSTM")
        
        return self.models
    
    def load_test_data(self, level='video'):
        """
        Carga datos de prueba para un nivel de agregación específico.
        
        Parámetros:
        -----------
        level : str
            Nivel de agregación ('video', 'object' o 'frame')
        """
        # Cargar datos de prueba
        test_data_path = os.path.join(self.data_dir, f'{level}_test_scaled.csv')
        if not os.path.exists(test_data_path):
            print(f"Error: No se encontraron datos de prueba para nivel '{level}'")
            return None
        
        test_data = pd.read_csv(test_data_path)
        
        # Cargar características seleccionadas
        features_path = os.path.join(self.models_dir, f'{level}_features.pkl')
        if os.path.exists(features_path):
            with open(features_path, 'rb') as f:
                features = pickle.load(f)
        else:
            # Si no se encuentran características guardadas, usar todas las numéricas
            exclude_cols = ['clase_encoded', 'video_id', 'Frame_', 'Objeto_', 'clase_first']
            features = [col for col in test_data.select_dtypes(include=['float64', 'int64']).columns
                      if not any(col.startswith(ex) for ex in exclude_cols)]
        
        # Preparar X e y
        X_test = test_data[features]
        y_test = test_data['clase_encoded']
        
        # Incluir información de clase original si está disponible
        class_info = None
        if 'clase_first' in test_data.columns:
            class_info = test_data[['clase_first', 'clase_encoded']].drop_duplicates()
        
        return {
            'X_test': X_test,
            'y_test': y_test,
            'features': features,
            'class_info': class_info,
            'test_data': test_data
        }
    
    def load_lstm_test_data(self):
        """
        Carga datos de prueba para el modelo LSTM.
        """
        # Cargar información de secuencias
        seq_info_path = os.path.join(self.data_dir, 'lstm_data', 'sequence_info.csv')
        if not os.path.exists(seq_info_path):
            print("Error: No se encontró información de secuencias para LSTM")
            return None
        
        seq_info = pd.read_csv(seq_info_path)
        
        # Cargar información de división
        split_info_path = os.path.join(self.data_dir, 'data_split.csv')
        if os.path.exists(split_info_path):
            split_info = pd.read_csv(split_info_path)
            seq_info = pd.merge(seq_info, split_info[['video_id', 'split']], on='video_id')
            test_info = seq_info[seq_info['split'] == 'test']
        else:
            # Si no hay información de división, usar todos los datos
            test_info = seq_info
        
        # Función para cargar secuencias
        def load_sequences(info_df, max_seq_len=None):
            sequences = []
            labels = []
            video_ids = []
            
            for _, row in info_df.iterrows():
                video_id = row['video_id']
                seq_file = os.path.join(self.data_dir, 'lstm_data', f'{video_id}.npy')
                
                if os.path.exists(seq_file):
                    seq = np.load(seq_file)
                    
                    # Normalizar longitud de secuencia si es necesario
                    if max_seq_len:
                        if len(seq) > max_seq_len:
                            # Truncar
                            seq = seq[:max_seq_len]
                        elif len(seq) < max_seq_len:
                            # Rellenar con ceros
                            padding = np.zeros((max_seq_len - len(seq), seq.shape[1]))
                            seq = np.vstack([seq, padding])
                    
                    sequences.append(seq)
                    labels.append(row['clase_encoded'])
                    video_ids.append(video_id)
            
            return sequences, labels, video_ids
        
        # Determinar longitud máxima de secuencia
        # Usar el mismo valor que en el entrenamiento si está disponible
        model_config_path = os.path.join(self.models_dir, 'lstm', 'model_config.npy')
        if os.path.exists(model_config_path):
            model_config = np.load(model_config_path, allow_pickle=True).item()
            max_seq_length = model_config.get('max_seq_length', 100)
        else:
            max_seq_length = 100  # Valor predeterminado
        
        # Cargar secuencias
        test_seq, test_labels, test_vids = load_sequences(test_info, max_seq_length)
        
        # Convertir a arrays numpy
        X_test = np.array(test_seq)
        y_test = np.array(test_labels)
        
        # Convertir etiquetas a one-hot
        num_classes = len(np.unique(y_test))
        y_test_onehot = tf.keras.utils.to_categorical(y_test, num_classes)
        
        return {
            'X_test': X_test,
            'y_test': y_test,
            'y_test_onehot': y_test_onehot,
            'video_ids': test_vids,
            'max_seq_length': max_seq_length,
            'num_classes': num_classes
        }
    
    def evaluate_model(self, model_type, level, test_data=None):
        """
        Evalúa un modelo específico con datos de prueba.
        
        Parámetros:
        -----------
        model_type : str
            Tipo de modelo ('mlp', 'rf', 'lstm')
        level : str
            Nivel de agregación ('video', 'object', 'frame')
        test_data : dict, optional
            Datos de prueba pre-cargados
        """
        model_name = f"{model_type}_{level}" if model_type != 'lstm' else 'lstm'
        
        # Verificar si el modelo existe
        if model_name not in self.models:
            print(f"Error: No se encontró el modelo '{model_name}'")
            return None
        
        model = self.models[model_name]
        
        # Cargar datos de prueba si no se proporcionaron
        if test_data is None:
            if model_type == 'lstm':
                test_data = self.load_lstm_test_data()
            else:
                test_data = self.load_test_data(level)
        
        if test_data is None:
            return None
        
        # Hacer predicciones
        X_test = test_data['X_test']
        y_test = test_data['y_test']
        
        if model_type == 'lstm':
            y_pred_proba = model.predict(X_test)
            y_pred = np.argmax(y_pred_proba, axis=1)
        else:
            y_pred = model.predict(X_test)
            
            # Calcular probabilidades si el modelo lo soporta
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)
            else:
                y_pred_proba = None
        
        # Calcular métricas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        
        # Reporte de clasificación
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Registrar resultados
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'classification_report': report,
            'y_true': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        # Guardar en resultados generales
        self.results[model_name] = results
        
        return results
    
    def evaluate_all_models(self):
        """
        Evalúa todos los modelos cargados en datos de prueba.
        """
        if not self.models:
            print("No hay modelos cargados. Ejecutando load_models() primero.")
            self.load_models()
        
        print("\nEvaluando modelos en datos de prueba...")
        
        # Evaluar modelos MLP y RF para cada nivel
        for level in ['video', 'object', 'frame']:
            # Cargar datos de prueba una vez para este nivel
            test_data = self.load_test_data(level)
            if not test_data:
                continue
            
            # Evaluar MLP si existe
            if f'mlp_{level}' in self.models:
                print(f"Evaluando modelo MLP para nivel '{level}'...")
                self.evaluate_model('mlp', level, test_data)
            
            # Evaluar Random Forest si existe
            if f'rf_{level}' in self.models:
                print(f"Evaluando modelo Random Forest para nivel '{level}'...")
                self.evaluate_model('rf', level, test_data)
        
        # Evaluar LSTM si existe
        if 'lstm' in self.models:
            print("Evaluando modelo LSTM...")
            self.evaluate_model('lstm', 'sequence')
        
        return self.results
    
    def plot_confusion_matrix(self, model_name):
        """
        Genera y guarda visualización de matriz de confusión para un modelo.
        
        Parámetros:
        -----------
        model_name : str
            Nombre del modelo a visualizar
        """
        if model_name not in self.results:
            print(f"Error: No se encontraron resultados para el modelo '{model_name}'")
            return
        
        # Obtener matriz de confusión
        cm = self.results[model_name]['confusion_matrix']
        
        # Determinar etiquetas de clase
        if any(x in model_name for x in ['mlp_', 'rf_']):
            level = model_name.split('_')[1]
            test_data = self.load_test_data(level)
            if test_data and test_data['class_info'] is not None:
                class_info = test_data['class_info']
                class_mapping = {
                    row['clase_encoded']: row['clase_first'] 
                    for _, row in class_info.iterrows()
                }
                labels = [class_mapping.get(i, str(i)) for i in range(len(cm))]
            else:
                labels = self.class_names[:len(cm)]
        else:
            # Para LSTM
            labels = self.class_names[:len(cm)]
        
        # Normalizar matriz
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Crear figura
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicción')
        plt.ylabel('Valor Real')
        plt.title(f'Matriz de Confusión - {model_name}')
        
        # Guardar figura
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'plots', f'{model_name}_confusion_matrix.png'))
        plt.close()
    
    def plot_roc_curves(self, model_name):
        """
        Genera y guarda curvas ROC para un modelo.
        
        Parámetros:
        -----------
        model_name : str
            Nombre del modelo a visualizar
        """
        if model_name not in self.results:
            print(f"Error: No se encontraron resultados para el modelo '{model_name}'")
            return
        
        results = self.results[model_name]
        
        # Verificar si hay probabilidades disponibles
        if results['y_pred_proba'] is None:
            print(f"No hay probabilidades disponibles para el modelo '{model_name}'")
            return
        
        y_true = results['y_true']
        y_pred_proba = results['y_pred_proba']
        
        # Determinar etiquetas de clase
        if any(x in model_name for x in ['mlp_', 'rf_']):
            level = model_name.split('_')[1]
            test_data = self.load_test_data(level)
            if test_data and test_data['class_info'] is not None:
                class_info = test_data['class_info']
                class_mapping = {
                    row['clase_encoded']: row['clase_first'] 
                    for _, row in class_info.iterrows()
                }
                class_names = [class_mapping.get(i, str(i)) for i in range(y_pred_proba.shape[1])]
            else:
                class_names = self.class_names[:y_pred_proba.shape[1]]
        else:
            # Para LSTM
            class_names = self.class_names[:y_pred_proba.shape[1]]
        
        # Calcular curvas ROC para cada clase
        fpr = {}
        tpr = {}
        roc_auc = {}
        
        for i in range(len(class_names)):
            # Convertir a one-hot para la clase actual
            y_true_bin = (y_true == i).astype(int)
            
            # Calcular curva ROC
            fpr[i], tpr[i], _ = roc_curve(y_true_bin, y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Graficar curvas ROC
        plt.figure(figsize=(10, 8))
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, color, cls in zip(range(len(class_names)), colors, class_names):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                    label=f'{cls} (AUC = {roc_auc[i]:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Tasa de Falsos Positivos')
        plt.ylabel('Tasa de Verdaderos Positivos')
        plt.title(f'Curvas ROC - {model_name}')
        plt.legend(loc="lower right")
        
        # Guardar figura
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'plots', f'{model_name}_roc_curves.png'))
        plt.close()
    
    def generate_performance_summary(self):
        """
        Genera y guarda un resumen del rendimiento de todos los modelos evaluados.
        """
        if not self.results:
            print("No hay resultados de evaluación. Ejecute evaluate_all_models() primero.")
            return
        
        # Recopilar métricas principales
        summary_data = []
        
        for model_name, results in self.results.items():
            model_type = model_name.split('_')[0] if '_' in model_name else model_name
            level = model_name.split('_')[1] if '_' in model_name else 'sequence'
            
            summary_data.append({
                'model_name': model_name,
                'model_type': model_type,
                'level': level,
                'accuracy': results['accuracy'],
                'precision': results['precision'],
                'recall': results['recall'],
                'f1': results['f1']
            })
        
        # Convertir a DataFrame
        summary_df = pd.DataFrame(summary_data)
        
        # Ordenar por F1-score
        summary_df.sort_values('f1', ascending=False, inplace=True)
        
        # Guardar resumen en CSV
        summary_df.to_csv(os.path.join(self.output_dir, 'performance_summary.csv'), index=False)
        
        # Crear visualización comparativa
        plt.figure(figsize=(12, 8))
        
        # Preparar datos para gráfico
        models = summary_df['model_name']
        accuracy = summary_df['accuracy']
        precision = summary_df['precision']
        recall = summary_df['recall']
        f1 = summary_df['f1']
        
        # Ancho de barras
        bar_width = 0.2
        index = np.arange(len(models))
        
        # Graficar barras
        plt.bar(index, accuracy, bar_width, label='Exactitud', color='b', alpha=0.7)
        plt.bar(index + bar_width, precision, bar_width, label='Precisión', color='g', alpha=0.7)
        plt.bar(index + 2*bar_width, recall, bar_width, label='Sensibilidad', color='r', alpha=0.7)
        plt.bar(index + 3*bar_width, f1, bar_width, label='F1-Score', color='purple', alpha=0.7)
        
        # Etiquetas y leyenda
        plt.xlabel('Modelo')
        plt.ylabel('Puntuación')
        plt.title('Comparación de Rendimiento de Modelos')
        plt.xticks(index + 1.5*bar_width, models, rotation=45, ha='right')
        plt.legend()
        
        # Ajustar layout y guardar
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'plots', 'model_comparison.png'))
        plt.close()
        
        # También crear gráfico específico por tipo de modelo
        for model_type in ['mlp', 'rf']:
            type_df = summary_df[summary_df['model_type'] == model_type]
            
            if len(type_df) > 0:
                plt.figure(figsize=(10, 6))
                
                # Preparar datos para gráfico
                levels = type_df['level']
                metrics = ['accuracy', 'precision', 'recall', 'f1']
                colors = ['blue', 'green', 'red', 'purple']
                
                # Crear gráfico de barras
                bar_width = 0.2
                index = np.arange(len(levels))
                
                for i, (metric, color) in enumerate(zip(metrics, colors)):
                    plt.bar(index + i*bar_width, type_df[metric], bar_width, 
                           label=metric.capitalize(), color=color, alpha=0.7)
                
                # Etiquetas y leyenda
                plt.xlabel('Nivel de Agregación')
                plt.ylabel('Puntuación')
                plt.title(f'Rendimiento del Modelo {model_type.upper()} por Nivel')
                plt.xticks(index + 1.5*bar_width, levels)
                plt.legend()
                
                # Ajustar layout y guardar
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, 'plots', f'{model_type}_comparison.png'))
                plt.close()
        
        return summary_df
    
    def generate_full_report(self):
        """
        Genera un informe completo de evaluación con visualizaciones para todos los modelos.
        """
        print("\nGenerando informe completo de evaluación...")
        
        # Si no hay modelos cargados, cargarlos
        if not self.models:
            self.load_models()
        
        # Si no hay resultados de evaluación, evaluar modelos
        if not self.results:
            self.evaluate_all_models()
        
        # Generar visualizaciones para cada modelo
        for model_name in self.results:
            print(f"Generando visualizaciones para modelo '{model_name}'...")
            
            # Matriz de confusión
            self.plot_confusion_matrix(model_name)
            
            # Curvas ROC
            self.plot_roc_curves(model_name)
        
        # Generar resumen de rendimiento
        summary = self.generate_performance_summary()
        
        # Generar informe HTML
        self.generate_html_report(summary)
        
        print(f"Informe completo generado en {self.output_dir}")
        
        return summary
    
    def generate_html_report(self, summary):
        """
        Genera un informe HTML con todos los resultados y visualizaciones.
        
        Parámetros:
        -----------
        summary : DataFrame
            Resumen de rendimiento de los modelos
        """
        # Crear contenido HTML
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Evaluación de Modelos de Clasificación de Comportamientos</title>
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
                .model-section {{ margin-bottom: 40px; border-bottom: 1px solid #ddd; padding-bottom: 20px; }}
                .footer {{ margin-top: 50px; text-align: center; color: #7f8c8d; font-size: 0.8em; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Evaluación de Modelos de Clasificación de Comportamientos</h1>
                <p>Fecha de evaluación: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}</p>
                
                <h2>Resumen de Rendimiento</h2>
                <table>
                    <tr>
                        <th>Modelo</th>
                        <th>Tipo</th>
                        <th>Nivel</th>
                        <th>Exactitud</th>
                        <th>Precisión</th>
                        <th>Sensibilidad</th>
                        <th>F1-Score</th>
                    </tr>
        """
        
        # Añadir filas de la tabla de resumen
        for _, row in summary.iterrows():
            html_content += f"""
                    <tr>
                        <td>{row['model_name']}</td>
                        <td>{row['model_type']}</td>
                        <td>{row['level']}</td>
                        <td>{row['accuracy']:.4f}</td>
                        <td>{row['precision']:.4f}</td>
                        <td>{row['recall']:.4f}</td>
                        <td>{row['f1']:.4f}</td>
                    </tr>
            """
        
        html_content += """
                </table>
                
                <div class="graph">
                    <h3>Comparación de Modelos</h3>
                    <img src="plots/model_comparison.png" alt="Comparación de Modelos">
                </div>
        """
        
        # Añadir gráficos de comparación por tipo de modelo
        for model_type in ['mlp', 'rf']:
            plot_path = os.path.join('plots', f'{model_type}_comparison.png')
            if os.path.exists(os.path.join(self.output_dir, plot_path)):
                html_content += f"""
                <div class="graph">
                    <h3>Comparación de Modelos {model_type.upper()}</h3>
                    <img src="{plot_path}" alt="Comparación de Modelos {model_type.upper()}">
                </div>
                """
        
        # Añadir secciones para cada modelo
        for model_name in self.results:
            results = self.results[model_name]
            
            # Rutas a visualizaciones
            cm_path = os.path.join('plots', f'{model_name}_confusion_matrix.png')
            roc_path = os.path.join('plots', f'{model_name}_roc_curves.png')
            
            html_content += f"""
                <div class="model-section">
                    <h2>Modelo: {model_name}</h2>
                    
                    <h3>Métricas</h3>
                    <table>
                        <tr>
                            <th>Métrica</th>
                            <th>Valor</th>
                        </tr>
                        <tr>
                            <td>Exactitud</td>
                            <td>{results['accuracy']:.4f}</td>
                        </tr>
                        <tr>
                            <td>Precisión</td>
                            <td>{results['precision']:.4f}</td>
                        </tr>
                        <tr>
                            <td>Sensibilidad</td>
                            <td>{results['recall']:.4f}</td>
                        </tr>
                        <tr>
                            <td>F1-Score</td>
                            <td>{results['f1']:.4f}</td>
                        </tr>
                    </table>
                    
                    <h3>Matriz de Confusión</h3>
                    <div class="graph">
                        <img src="{cm_path}" alt="Matriz de Confusión - {model_name}">
                    </div>
            """
            
            # Añadir curvas ROC si existen
            if os.path.exists(os.path.join(self.output_dir, roc_path)):
                html_content += f"""
                    <h3>Curvas ROC</h3>
                    <div class="graph">
                        <img src="{roc_path}" alt="Curvas ROC - {model_name}">
                    </div>
                """
            
            html_content += """
                </div>
            """
        
        # Cerrar documento HTML
        html_content += """
                <div class="footer">
                    <p>Generado automáticamente por ModelEvaluator</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Guardar informe HTML
        report_path = os.path.join(self.output_dir, 'evaluation_report.html')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

if __name__ == "__main__":
    evaluator = ModelEvaluator()
    evaluator.generate_full_report()