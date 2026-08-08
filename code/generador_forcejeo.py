"""
⚠️ DATOS SINTÉTICOS — NO son extraídos de video real ⚠️

Este módulo NO analiza video. Genera filas de CSV con la misma estructura
que produce featureExtraction.py, pero los valores (trayectorias, velocidad,
linealidad, ciclicidad, etc.) se generan por muestreo estadístico
(distribuciones normal/exponencial/gamma/beta) con una semilla fija, no a
partir de comportamiento real de merodeo observado en cámara.

Antes de usar generate_all_datasets() para ampliar el set de entrenamiento,
ten en cuenta que:
- El modelo puede terminar aprendiendo las asunciones estadísticas de este
  generador en lugar de patrones reales de comportamiento sospechoso.
- Cualquier archivo *_procesado.csv generado aquí debe quedar identificado
  como sintético (por nombre de archivo o en un registro aparte) y nunca
  mezclarse sin marcar con el set de validación/test, para no inflar las
  métricas de forma artificial.
- Si el protocolo del TT no contempla explícitamente aumentación de datos
  sintética, conviene confirmarlo antes de reportar métricas entrenadas con
  estos archivos.
"""

import pandas as pd
import numpy as np
import random
import math
from typing import List, Dict, Tuple

class MerodeoDataSimulator:
    def __init__(self, seed: int = 42):
        """Inicializa el simulador con parámetros basados en el análisis de datos reales."""
        np.random.seed(seed)
        random.seed(seed)
        
        # Rangos observados en los datos reales
        self.frame_ranges = {
            'small': (30, 500),    # archivos pequeños
            'medium': (500, 7000), # archivos medianos  
            'large': (500, 20000)  # archivos grandes
        }
        
        self.centroid_ranges = {
            'x': (17, 450),
            'y': (20, 300)
        }
        
        self.movement_patterns = {
            0: 0.85,  # Patrón 0 es dominante (85%)
            1: 0.02,  # Patrón 1 (2%)
            2: 0.08,  # Patrón 2 (8%)
            3: 0.03,  # Patrón 3 (3%)
            4: 0.02   # Patrón 4 (2%)
        }
        
        self.posture_dist = [0.1, 0.2, 0.7]  # [0, 1, 2] distribución de posturas
        
    def generate_trajectory_metrics(self, num_points: int) -> Dict:
        """Genera métricas de trayectoria realistas."""
        # Generar trayectoria base
        angles = np.cumsum(np.random.normal(0, 0.5, num_points))
        distances = np.abs(np.random.normal(25, 15, num_points))
        
        # Calcular métricas
        total_distance = np.sum(distances)
        displacement_vector = np.sum([d * np.array([np.cos(a), np.sin(a)]) 
                                     for d, a in zip(distances, angles)], axis=0)
        net_displacement = np.linalg.norm(displacement_vector)
        
        linealidad = net_displacement / total_distance if total_distance > 0 else 0
        
        # Circularidad - basada en cuán circular es la trayectoria
        circularidad = max(0, min(1, np.random.beta(2, 3)))
        
        # Zigzag - basada en cambios de dirección
        direction_changes = np.sum(np.abs(np.diff(angles)) > 0.5)
        zigzag = min(1, direction_changes / max(1, num_points - 1))
        
        # Área de trayectoria (aproximada)
        area = max(0, np.random.exponential(5000))
        
        return {
            'linealidad': linealidad,
            'circularidad': circularidad, 
            'zigzag': zigzag,
            'area_trayectoria': area
        }
    
    def generate_cyclic_behavior(self) -> Dict:
        """Genera comportamiento cíclico."""
        is_cyclic = np.random.choice([0, 1], p=[0.2, 0.8])  # 80% son cíclicos
        
        if is_cyclic:
            frequency = np.random.uniform(0.01, 0.5)
            amplitude = np.random.uniform(5, 90)
        else:
            frequency = 0
            amplitude = 0
            
        return {
            'es_ciclico': is_cyclic,
            'frecuencia_ciclo': frequency,
            'amplitud_ciclo': amplitude
        }
    
    def generate_object_trajectory(self, object_id: str, start_frame: int, 
                                 num_frames: int, start_pos: Tuple[int, int]) -> List[Dict]:
        """Genera la trayectoria completa de un objeto."""
        trajectory = []
        
        # Generar métricas de trayectoria
        traj_metrics = self.generate_trajectory_metrics(num_frames)
        cyclic_behavior = self.generate_cyclic_behavior()
        
        # Patrón de movimiento para todo el objeto
        movement_pattern = np.random.choice(list(self.movement_patterns.keys()), 
                                          p=list(self.movement_patterns.values()))
        
        # Postura (puede cambiar durante la trayectoria)
        base_posture = np.random.choice([0, 1, 2], p=self.posture_dist)
        
        # Generar nombre del objeto original
        obj_types = ['obj_', 'movimiento_']
        obj_type = np.random.choice(obj_types)
        obj_original = f"{obj_type}{np.random.randint(1, 10000)}"
        
        # Variables de estado
        current_x, current_y = start_pos
        prev_velocity = 0
        prev_direction = 0
        
        for i in range(num_frames):
            frame = start_frame + i * np.random.randint(1, 5)  # Frames no consecutivos
            
            # Movimiento
            if i == 0:
                # Primer frame - sin movimiento
                displacement = 0
                velocity = 0
                acceleration = 0
                direction = 0
            else:
                # Generar desplazamiento basado en patrón de movimiento
                if movement_pattern == 0:  # Movimiento básico
                    base_disp = np.random.exponential(20)
                elif movement_pattern == 1:  # Movimiento lineal
                    base_disp = np.random.uniform(10, 50)
                elif movement_pattern == 2:  # Movimiento circular
                    base_disp = np.random.uniform(5, 30)
                elif movement_pattern == 3:  # Movimiento errático
                    base_disp = np.random.exponential(40)
                else:  # movement_pattern == 4, movimiento complejo
                    base_disp = np.random.gamma(2, 15)
                
                displacement = min(base_disp, 300)  # Limitar desplazamiento máximo
                
                # Dirección
                if cyclic_behavior['es_ciclico']:
                    # Movimiento cíclico
                    cycle_phase = 2 * math.pi * cyclic_behavior['frecuencia_ciclo'] * i
                    direction = prev_direction + 30 * math.sin(cycle_phase)
                else:
                    # Movimiento más aleatorio
                    direction = prev_direction + np.random.normal(0, 45)
                
                direction = max(-180, min(180, direction))
                
                # Velocidad (desplazamiento * factor de conversión temporal)
                time_factor = np.random.uniform(20, 30)  # fps simulado
                velocity = displacement * time_factor
                
                # Aceleración
                acceleration = (velocity - prev_velocity) * time_factor
                
                # Actualizar posición
                rad_direction = math.radians(direction)
                current_x += displacement * math.cos(rad_direction)
                current_y += displacement * math.sin(rad_direction)
                
                # Mantener dentro de límites
                current_x = max(self.centroid_ranges['x'][0], 
                              min(self.centroid_ranges['x'][1], current_x))
                current_y = max(self.centroid_ranges['y'][0], 
                              min(self.centroid_ranges['y'][1], current_y))
                
                prev_velocity = velocity
                prev_direction = direction
            
            # Densidad (varía ligeramente)
            density = np.random.uniform(0.0007, 0.0055)
            
            # Postura (puede cambiar ocasionalmente)
            if np.random.random() < 0.05:  # 5% probabilidad de cambio
                posture = np.random.choice([0, 1, 2], p=self.posture_dist)
            else:
                posture = base_posture
            
            # Interacción (muy raro en los datos originales)
            interaction = 0  # Siempre 0 en los datos originales
            
            row = {
                'Frame': frame,
                'Objeto': object_id,
                'Centroide_X': int(current_x),
                'Centroide_Y': int(current_y),
                'Desplazamiento': displacement,
                'Velocidad': velocity,
                'Aceleracion': acceleration,
                'Direccion': direction,
                'Densidad': density,
                'Postura': posture,
                'Patron_Movimiento': movement_pattern,
                'Linealidad': traj_metrics['linealidad'],
                'Circularidad': traj_metrics['circularidad'],
                'Zigzag': traj_metrics['zigzag'],
                'Es_Ciclico': cyclic_behavior['es_ciclico'],
                'Frecuencia_Ciclo': cyclic_behavior['frecuencia_ciclo'],
                'Amplitud_Ciclo': cyclic_behavior['amplitud_ciclo'],
                'Area_Trayectoria': traj_metrics['area_trayectoria'],
                'En_Interaccion': interaction,
                'Objeto_Original': obj_original
            }
            
            trajectory.append(row)
        
        return trajectory
    
    def generate_dataset(self, target_rows: int, file_category: str) -> pd.DataFrame:
        """Genera un dataset completo con el número objetivo de filas."""
        
        # Determinar rango de frames según categoría
        if file_category == 'small':
            frame_range = self.frame_ranges['small']
            avg_traj_length = np.random.randint(5, 25)
        elif file_category == 'medium':
            frame_range = self.frame_ranges['medium']  
            avg_traj_length = np.random.randint(15, 80)
        else:  # large
            frame_range = self.frame_ranges['large']
            avg_traj_length = np.random.randint(30, 150)
        
        # Calcular número aproximado de objetos necesarios
        estimated_objects = max(1, target_rows // avg_traj_length)
        
        all_data = []
        current_rows = 0
        object_counter = 0
        
        while current_rows < target_rows and object_counter < estimated_objects * 2:
            # Generar parámetros para este objeto
            remaining_rows = target_rows - current_rows
            traj_length = min(remaining_rows, 
                            np.random.poisson(avg_traj_length) + 1)
            
            if traj_length <= 0:
                break
                
            start_frame = np.random.randint(frame_range[0], 
                                          max(frame_range[0] + 1, frame_range[1] - traj_length * 5))
            
            start_pos = (np.random.randint(self.centroid_ranges['x'][0], 
                                         self.centroid_ranges['x'][1]),
                        np.random.randint(self.centroid_ranges['y'][0], 
                                        self.centroid_ranges['y'][1]))
            
            object_id = f"objeto_real_{object_counter}"
            
            # Generar trayectoria
            trajectory = self.generate_object_trajectory(object_id, start_frame, 
                                                      traj_length, start_pos)
            
            all_data.extend(trajectory)
            current_rows += len(trajectory)
            object_counter += 1
        
        # Crear DataFrame y ordenar por Frame
        df = pd.DataFrame(all_data[:target_rows])  # Cortar a tamaño exacto
        df = df.sort_values('Frame').reset_index(drop=True)
        
        return df

def generate_all_datasets():
    """Genera 8 nuevos datasets de merodeo con características similares a los originales."""
    
    simulator = MerodeoDataSimulator()
    
    # Definir los 8 nuevos archivos con tamaños variados basados en la distribución original
    new_files = [
        {'name': 'merodeo_45_procesado.csv', 'rows': 156, 'category': 'small'},    # Similar a merodeo_32 
        {'name': 'merodeo_47_procesado.csv', 'rows': 423, 'category': 'small'},    # Similar a merodeo_6
        {'name': 'merodeo_50_procesado.csv', 'rows': 1456, 'category': 'medium'},  # Similar a merodeo_9
        {'name': 'merodeo_53_procesado.csv', 'rows': 3891, 'category': 'medium'},  # Similar a merodeo_24
        {'name': 'merodeo_56_procesado.csv', 'rows': 5234, 'category': 'medium'},  # Similar a merodeo_1
        {'name': 'merodeo_59_procesado.csv', 'rows': 7832, 'category': 'medium'},  # Entre medianos y grandes
        {'name': 'merodeo_61_procesado.csv', 'rows': 9156, 'category': 'large'},   # Similar a merodeo_42
        {'name': 'merodeo_64_procesado.csv', 'rows': 14278, 'category': 'large'}   # Similar a merodeo_18
    ]
    
    print("Generando 8 nuevos datasets de merodeo...")
    print("="*60)
    
    for file_info in new_files:
        print(f"\nGenerando {file_info['name']}...")
        print(f"  - Filas objetivo: {file_info['rows']}")
        print(f"  - Categoría: {file_info['category']}")
        
        # Generar dataset
        df = simulator.generate_dataset(file_info['rows'], file_info['category'])
        
        # Guardar archivo
        df.to_csv(file_info['name'], index=False)
        
        print(f"  - Generado exitosamente: {len(df)} filas, {df['Objeto'].nunique()} objetos únicos")
        print(f"  - Rango de frames: {df['Frame'].min()} - {df['Frame'].max()}")
        
        # Mostrar estadísticas básicas
        pattern_counts = df['Patron_Movimiento'].value_counts().sort_index()
        print(f"  - Patrones de movimiento: {dict(pattern_counts)}")
    
    print("\n" + "="*60)
    print("¡Generación completada! Se han creado 8 nuevos archivos CSV.")
    print("\nCaracterísticas de los archivos generados:")
    print("- Misma estructura de 20 columnas que los archivos originales")
    print("- Distribución similar de patrones de movimiento")  
    print("- Rangos de valores consistentes con los datos reales")
    print("- Trayectorias realistas con comportamientos cíclicos")
    print("- Métricas de movimiento calculadas apropiadamente")

# Ejecutar la generación
if __name__ == "__main__":
    generate_all_datasets()