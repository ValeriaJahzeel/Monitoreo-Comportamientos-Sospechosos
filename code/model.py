import os
import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm  # Para barra de progreso
from torch.optim.lr_scheduler import ReduceLROnPlateau  # Scheduler de tasa de aprendizaje
import time
import logging

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training_log.txt"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Función para memoria de GPU
def print_gpu_memory(message=""):
    if torch.cuda.is_available():
        t = torch.cuda.get_device_properties(0).total_memory / 1e9
        r = torch.cuda.memory_reserved(0) / 1e9
        a = torch.cuda.memory_allocated(0) / 1e9
        f = t - (r + a)
        logger.info(f"{message} GPU Memory: Total {t:.2f}GB | Reserved {r:.2f}GB | Allocated {a:.2f}GB | Free {f:.2f}GB")

class VideoFrameDataset(Dataset):
    def __init__(self, csv_dir, selected_features=None, max_frames=None, normalize=True, cache_data=True):
        """
        Conjunto de datos para clasificación de videos con longitudes variables
        
        Args:
            csv_dir (str): Directorio con archivos CSV de videos
            selected_features (list, optional): Lista de características a usar
            max_frames (int, optional): Número máximo de frames a considerar
            normalize (bool): Si se debe normalizar los datos
            cache_data (bool): Si se debe almacenar en caché los datos para acceso más rápido
        """
        self.videos = []
        self.labels = []
        self.video_lengths = []
        self.cache_data = cache_data
        self.data_cache = {}
        
        logger.info(f"Cargando datos de {csv_dir}...")
        start_time = time.time()
        
        # Verificar que el directorio existe
        if not os.path.exists(csv_dir):
            raise ValueError(f"El directorio {csv_dir} no existe")
        
        csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
        if len(csv_files) == 0:
            raise ValueError(f"No se encontraron archivos CSV en {csv_dir}")
        
        # Procesamiento en paralelo para archivos grandes
        normal_count = 0
        suspicious_count = 0
        
        # Mapeo de etiquetas más flexible
        for filename in tqdm(os.listdir(csv_dir), desc="Cargando videos"):
            if filename.endswith('.csv'):
                filepath = os.path.join(csv_dir, filename)
                
                # Leer CSV con encabezados y manejo de errores
                try:
                    df = pd.read_csv(filepath)
                except Exception as e:
                    logger.error(f"Error al leer {filename}: {str(e)}")
                    continue
                
                # Selección de características
                if selected_features is None:
                    # Si no se especifican, usar todas las características numéricas
                    features = df.select_dtypes(include=[np.number])
                else:
                    # Verificar que las características existen
                    missing_features = [f for f in selected_features if f not in df.columns]
                    if missing_features:
                        logger.warning(f"Características no encontradas en {filename}: {missing_features}")
                        # Usar solo las características que sí existen
                        valid_features = [f for f in selected_features if f in df.columns]
                        if not valid_features:
                            logger.error(f"No se encontraron características válidas en {filename}")
                            continue
                        features = df[valid_features]
                    else:
                        # Usar solo las características especificadas
                        features = df[selected_features]
                
                # Convertir a numpy array con manejo de NaN
                features_array = features.values
                if np.isnan(features_array).any():
                    logger.warning(f"Se encontraron valores NaN en {filename}, rellenando con 0")
                    features_array = np.nan_to_num(features_array)
                
                # Limitar número de frames si se especifica
                if max_frames is not None and len(features_array) > max_frames:
                    features_array = features_array[:max_frames]
                
                # Mapeo de etiquetas basado en prefijo del nombre del archivo
                if filename.startswith('normal_'):
                    label = 0
                    normal_count += 1
                elif filename.startswith('sospechoso_'):
                    label = 1
                    suspicious_count += 1
                else:
                    logger.warning(f"Archivo {filename} ignorado - etiqueta no reconocida")
                    continue
                
                # Normalización de características del video
                if normalize:
                    scaler = StandardScaler()
                    try:
                        normalized_features = scaler.fit_transform(features_array)
                    except Exception as e:
                        logger.error(f"Error al normalizar {filename}: {str(e)}")
                        continue
                else:
                    normalized_features = features_array
                
                # Convertir a tensor de PyTorch
                video_tensor = torch.FloatTensor(normalized_features)
                
                self.videos.append(video_tensor)
                self.labels.append(label)
                self.video_lengths.append(len(video_tensor))
        
        # Verificar que se hayan cargado videos
        if not self.videos:
            raise ValueError("No se encontraron videos válidos. Verifica tus archivos CSV.")
        
        elapsed_time = time.time() - start_time
        logger.info(f"Total de videos cargados: {len(self.videos)} en {elapsed_time:.2f} segundos")
        logger.info(f"Videos normales: {normal_count}")
        logger.info(f"Videos sospechosos: {suspicious_count}")
        logger.info(f"Longitudes de videos - Min: {min(self.video_lengths)}, Max: {max(self.video_lengths)}, Promedio: {np.mean(self.video_lengths):.2f}")
        
        # Calcular dimensiones para realizar validaciones
        if len(self.videos) > 0:
            self.input_dim = self.videos[0].shape[1]
            logger.info(f"Dimensión de características: {self.input_dim}")
    
    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, idx):
        if self.cache_data and idx in self.data_cache:
            return self.data_cache[idx]
        
        item = (self.videos[idx], torch.LongTensor([self.labels[idx]]))
        
        if self.cache_data:
            self.data_cache[idx] = item
            
        return item

def collate_fn(batch):
    """
    Función personalizada para manejar lotes con longitudes variables
    """
    videos, labels = zip(*batch)
    
    # Ordenar por longitud para optimizar el padding
    videos_sorted, labels_sorted = zip(*sorted(zip(videos, labels), 
                                             key=lambda x: x[0].shape[0], 
                                             reverse=True))
    
    padded_videos = pad_sequence(videos_sorted, batch_first=True)
    labels_tensor = torch.cat(labels_sorted)
    
    # También devolver las longitudes para posibles masking
    lengths = torch.LongTensor([len(x) for x in videos_sorted])
    
    return padded_videos, labels_tensor, lengths

class VideoClassificationLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_rate=0.3, bidirectional=True):
        super(VideoClassificationLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.directions = 2 if bidirectional else 1
        
        # LSTM para secuencias
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Atención para dar mayor peso a frames importantes
        self.attention = nn.Linear(hidden_size * self.directions, 1)
        
        # Clasificador
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * self.directions, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_classes)
        )
        
        # Inicialización de pesos para convergencia más rápida
        self._initialize_weights()
    
    def _initialize_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, x, lengths=None):
        batch_size, seq_len, _ = x.size()
        
        # Si se proporcionan longitudes, usarlas para empaquetar
        if lengths is not None:
            # Empaquetar secuencia para computación eficiente
            packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
            packed_output, (hidden, _) = self.lstm(packed_input)
            # Desempaquetar
            output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        else:
            output, (hidden, _) = self.lstm(x)
        
        # Mecanismo de atención
        attention_weights = torch.softmax(self.attention(output).squeeze(-1), dim=1)
        context = torch.bmm(attention_weights.unsqueeze(1), output).squeeze(1)
        
        # Clasificación
        logits = self.fc(context)
        
        return logits

class VideoClassificationMLP(nn.Module):
    def __init__(self, input_size, hidden_layers, num_classes, dropout_rate=0.3):
        super(VideoClassificationMLP, self).__init__()
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))  # Normalización por lotes
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x, lengths=None):
        # Convertir (batch, seq_len, features) a (batch, features, seq_len)
        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(-1)
        
        return self.model(x)

def train_one_epoch(model, train_loader, criterion, optimizer, device, use_lengths=True):
    """
    Entrenar el modelo por una época
    """
    model.train()
    train_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    
    # Utilizar tqdm para mostrar progreso
    for batch_videos, batch_labels, lengths in tqdm(train_loader, desc="Entrenando"):
        batch_videos = batch_videos.to(device)
        batch_labels = batch_labels.to(device)
        
        optimizer.zero_grad()
        
        if use_lengths:
            outputs = model(batch_videos, lengths)
        else:
            outputs = model(batch_videos)
        
        loss = criterion(outputs, batch_labels)
        
        loss.backward()
        
        # Clipeo de gradiente para evitar explosión
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        train_loss += loss.item()
        
        _, predicted = torch.max(outputs, 1)
        total_predictions += batch_labels.size(0)
        correct_predictions += (predicted == batch_labels).sum().item()
    
    # Métricas de entrenamiento
    train_loss_avg = train_loss / len(train_loader)
    train_accuracy = 100 * correct_predictions / total_predictions
    
    return train_loss_avg, train_accuracy

def validate(model, val_loader, criterion, device, use_lengths=True):
    """
    Validar el modelo
    """
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch_videos, batch_labels, lengths in tqdm(val_loader, desc="Validando"):
            batch_videos = batch_videos.to(device)
            batch_labels = batch_labels.to(device)
            
            if use_lengths:
                outputs = model(batch_videos, lengths)
            else:
                outputs = model(batch_videos)
                
            loss = criterion(outputs, batch_labels)
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            val_total += batch_labels.size(0)
            val_correct += (predicted == batch_labels).sum().item()
            
            # Guardar para métricas adicionales
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
    
    # Calcular métricas
    val_loss_avg = val_loss / len(val_loader)
    val_accuracy = 100 * val_correct / val_total
    
    return val_loss_avg, val_accuracy, all_predictions, all_labels

def train_video_classifier(csv_dir, 
                           model_type='lstm',  # 'lstm' o 'mlp'
                           selected_features=None,
                           max_frames=None,
                           hidden_size=64,     # Para LSTM
                           num_layers=2,       # Para LSTM
                           hidden_layers=[64, 32],  # Para MLP
                           learning_rate=0.001, 
                           epochs=100, 
                           batch_size=16,
                           dropout_rate=0.3,
                           bidirectional=True,  # Para LSTM
                           patience=10,         # Early stopping
                           weight_decay=1e-5):  # Regularización L2
    """
    Entrenar clasificador de videos
    """
    start_time = time.time()
    
    # Determinar el dispositivo disponible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Utilizando dispositivo: {device}")
    
    if torch.cuda.is_available():
        # Establecer semilla para reproducibilidad en GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Cargar datos
    logger.info("Cargando conjunto de datos...")
    dataset = VideoFrameDataset(csv_dir, selected_features, max_frames)
    
    # Verificar si hay suficientes muestras para la estratificación
    min_samples_per_class = min(dataset.labels.count(0), dataset.labels.count(1))
    if min_samples_per_class < 2:
        logger.warning(f"Pocas muestras para estratificación. Clases: 0={dataset.labels.count(0)}, 1={dataset.labels.count(1)}")
        stratify = None
    else:
        stratify = dataset.labels
    
    # Dividir datos
    train_indices, val_indices = train_test_split(
        range(len(dataset)), test_size=0.2, stratify=stratify, random_state=42
    )
    
    # Crear subconjuntos
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    logger.info(f"Tamaño del conjunto de entrenamiento: {len(train_dataset)}")
    logger.info(f"Tamaño del conjunto de validación: {len(val_dataset)}")
    
    # Crear DataLoaders con trabajadores en paralelo
    num_workers = min(4, os.cpu_count() or 1)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print_gpu_memory("Antes de crear el modelo")
    
    # Configurar modelo
    input_size = dataset.videos[0].shape[1]
    num_classes = 2
    
    # Seleccionar arquitectura
    if model_type.lower() == 'lstm':
        logger.info(f"Creando modelo LSTM con {hidden_size} unidades, {num_layers} capas, bidireccional={bidirectional}")
        model = VideoClassificationLSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            num_classes=num_classes, 
            dropout_rate=dropout_rate,
            bidirectional=bidirectional
        ).to(device)
        use_lengths = True
    else:
        logger.info(f"Creando modelo MLP con capas ocultas {hidden_layers}")
        model = VideoClassificationMLP(
            input_size=input_size, 
            hidden_layers=hidden_layers, 
            num_classes=num_classes, 
            dropout_rate=dropout_rate
        ).to(device)
        use_lengths = False
    
    # Contar y loggear parámetros del modelo
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total de parámetros: {total_params:,}")
    logger.info(f"Parámetros entrenables: {trainable_params:,}")
    
    print_gpu_memory("Después de crear el modelo")
    
    # Determinar pesos para clases desbalanceadas
    if dataset.labels.count(0) != dataset.labels.count(1):
        class_counts = [dataset.labels.count(0), dataset.labels.count(1)]
        weights = torch.FloatTensor([len(dataset) / (2 * count) for count in class_counts]).to(device)
        logger.info(f"Usando pesos para clases desbalanceadas: {weights}")
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Optimizador con decaimiento de pesos para regularización
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Scheduler para reducir la tasa de aprendizaje cuando la pérdida se estanca
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Variables para early stopping
    best_val_accuracy = 0
    best_val_loss = float('inf')
    early_stopping_counter = 0
    
    # Para registrar historial de métricas
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': []
    }
    
    # Ciclo de entrenamiento
    logger.info(f"Comenzando entrenamiento por {epochs} épocas...")
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        # Entrenar una época
        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, criterion, optimizer, device, use_lengths
        )
        
        # Validar
        val_loss, val_accuracy, all_predictions, all_labels = validate(
            model, val_loader, criterion, device, use_lengths
        )
        
        # Actualizar learning rate basado en pérdida de validación
        scheduler.step(val_loss)
        
        # Guardar métricas
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_accuracy)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        
        epoch_time = time.time() - epoch_start_time
        
        # Mostrar progreso
        logger.info(f"Época {epoch+1}/{epochs} - "
                  f"Tiempo: {epoch_time:.2f}s - "
                  f"Train Loss: {train_loss:.4f} - "
                  f"Train Acc: {train_accuracy:.2f}% - "
                  f"Val Loss: {val_loss:.4f} - "
                  f"Val Acc: {val_accuracy:.2f}% - "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        print_gpu_memory(f"Después de época {epoch+1}")
        
        # Verificar si es el mejor modelo
        if val_accuracy > best_val_accuracy:
            logger.info(f"¡Nueva mejor precisión! {best_val_accuracy:.2f}% -> {val_accuracy:.2f}%")
            best_val_accuracy = val_accuracy
            best_val_loss = val_loss
            
            # Guardar el mejor modelo
            model_filename = f"best_model_{model_type}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_accuracy,
                'val_loss': val_loss,
                'train_accuracy': train_accuracy,
                'train_loss': train_loss,
                'model_type': model_type,
                'input_size': input_size,
                'hidden_size': hidden_size if model_type.lower() == 'lstm' else None,
                'num_layers': num_layers if model_type.lower() == 'lstm' else None,
                'hidden_layers': hidden_layers if model_type.lower() == 'mlp' else None,
                'bidirectional': bidirectional if model_type.lower() == 'lstm' else None,
            }, model_filename)
            
            logger.info(f"Modelo guardado en {model_filename}")
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            logger.info(f"No hay mejora en el rendimiento. Counter: {early_stopping_counter}/{patience}")
        
        # Early stopping
        # if early_stopping_counter >= patience:
        #     logger.info(f"Early stopping después de {epoch+1} épocas")
        #     break
    
    total_time = time.time() - start_time
    logger.info(f"Entrenamiento completado en {total_time/60:.2f} minutos")
    
    # Visualizar progreso de entrenamiento
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Entrenamiento')
    plt.plot(history['val_loss'], label='Validación')
    plt.title('Pérdida durante entrenamiento')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_accuracy'], label='Entrenamiento')
    plt.plot(history['val_accuracy'], label='Validación')
    plt.title('Precisión durante entrenamiento')
    plt.xlabel('Época')
    plt.ylabel('Precisión (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'training_history_{model_type}.png')
    plt.close()
    
    logger.info(f"Gráfico de entrenamiento guardado como training_history_{model_type}.png")
    
    # Cargar el mejor modelo para evaluación final
    checkpoint = torch.load(f"best_model_{model_type}.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluación final
    _, final_val_accuracy, _, _ = validate(model, val_loader, criterion, device, use_lengths)
    logger.info(f"Precisión final del mejor modelo: {final_val_accuracy:.2f}%")
    
    return best_val_accuracy, history

def grid_search(csv_dir, 
                model_types=['lstm', 'mlp'],
                hidden_sizes=[64, 128],           # Para LSTM
                num_layers_options=[1, 2],        # Para LSTM
                hidden_layers_options=[[64, 32], [128, 64, 32], [64, 64]],  # Para MLP
                learning_rates=[0.001, 0.0001],
                batch_sizes=[16, 32],
                dropout_rates=[0.3, 0.5],
                bidirectional_options=[True, False]):  # Para LSTM
    """
    Realizar grid search sobre hiperparámetros
    """
    logger.info("Iniciando grid search...")
    
    # Almacenar resultados del grid search
    results = []
    
    # Iterar sobre tipos de modelo
    for model_type in model_types:
        if model_type.lower() == 'lstm':
            # Generar combinaciones para LSTM
            hyperparameter_combinations = list(itertools.product(
                hidden_sizes,
                num_layers_options,
                learning_rates,
                batch_sizes,
                dropout_rates,
                bidirectional_options
            ))
            
            # Iterar sobre combinaciones de hiperparámetros
            for hidden_size, num_layers, lr, batch_size, dropout_rate, bidirectional in hyperparameter_combinations:
                logger.info("\n" + "="*50)
                logger.info(f"Entrenando LSTM con:")
                logger.info(f"Hidden Size: {hidden_size}")
                logger.info(f"Num Layers: {num_layers}")
                logger.info(f"Learning Rate: {lr}")
                logger.info(f"Batch Size: {batch_size}")
                logger.info(f"Dropout Rate: {dropout_rate}")
                logger.info(f"Bidirectional: {bidirectional}")
                
                try:
                    val_accuracy, _ = train_video_classifier(
                        csv_dir=csv_dir,
                        model_type='lstm',
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        learning_rate=lr,
                        batch_size=batch_size,
                        dropout_rate=dropout_rate,
                        bidirectional=bidirectional,
                        epochs=50,  # Reducir épocas para grid search
                        patience=5,  # Reducir paciencia para grid search
                        max_frames=10000
                    )
                    
                    results.append({
                        'model_type': 'lstm',
                        'hidden_size': hidden_size,
                        'num_layers': num_layers,
                        'hidden_layers': None,
                        'learning_rate': lr,
                        'batch_size': batch_size,
                        'dropout_rate': dropout_rate,
                        'bidirectional': bidirectional,
                        'val_accuracy': val_accuracy
                    })
                except Exception as e:
                    logger.error(f"Error durante entrenamiento: {str(e)}")
        
        elif model_type.lower() == 'mlp':
            # Generar combinaciones para MLP
            hyperparameter_combinations = list(itertools.product(
                hidden_layers_options,
                learning_rates,
                batch_sizes,
                dropout_rates
            ))
            
            # Iterar sobre combinaciones de hiperparámetros
            for hidden_layers, lr, batch_size, dropout_rate in hyperparameter_combinations:
                logger.info("\n" + "="*50)
                logger.info(f"Entrenando MLP con:")
                logger.info(f"Capas ocultas: {hidden_layers}")
                logger.info(f"Tasa de aprendizaje: {lr}")
                logger.info(f"Tamaño de lote: {batch_size}")
                logger.info(f"Tasa de dropout: {dropout_rate}")
                
                try:
                    val_accuracy, _ = train_video_classifier(
                        csv_dir=csv_dir,
                        model_type='mlp',
                        hidden_layers=hidden_layers,
                        learning_rate=lr,
                        batch_size=batch_size,
                        dropout_rate=dropout_rate,
                        epochs=50,  # Reducir épocas para grid search
                        patience=5,  # Reducir paciencia para grid search
                        max_frames=10000
                    )
                    
                    results.append({
                        'model_type': 'mlp',
                        'hidden_size': None,
                        'num_layers': None,
                        'hidden_layers': str(hidden_layers),
                        'bidirectional': None,
                        'val_accuracy': val_accuracy
                    })
                except Exception as e:
                    logger.error(f"Error durante entrenamiento: {str(e)}")
    
    # Ordenar resultados por precisión de validación
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('val_accuracy', ascending=False)
    
    # Guardar resultados
    results_df.to_csv('grid_search_results.csv', index=False)
    logger.info("\nMejores resultados:")
    for i, row in results_df.head().iterrows():
        if row['model_type'] == 'lstm':
            logger.info(f"LSTM - Val Acc: {row['val_accuracy']:.2f}% - Hidden: {row['hidden_size']} - "
                      f"Layers: {row['num_layers']} - LR: {row['learning_rate']} - "
                      f"Batch: {row['batch_size']} - Dropout: {row['dropout_rate']} - "
                      f"Bidirectional: {row['bidirectional']}")
        else:
            logger.info(f"MLP - Val Acc: {row['val_accuracy']:.2f}% - Hidden Layers: {row['hidden_layers']} - "
                      f"LR: {row['learning_rate']} - Batch: {row['batch_size']} - "
                      f"Dropout: {row['dropout_rate']}")
    
    # Visualizar resultados
    plt.figure(figsize=(12, 6))
    
    # Filtrar resultados por tipo de modelo
    lstm_results = results_df[results_df['model_type'] == 'lstm']
    mlp_results = results_df[results_df['model_type'] == 'mlp']
    
    if not lstm_results.empty:
        plt.subplot(1, 2, 1)
        plt.title('Resultados LSTM')
        plt.bar(range(min(5, len(lstm_results))), lstm_results['val_accuracy'].head(5))
        plt.xticks(range(min(5, len(lstm_results))), [f"Config {i+1}" for i in range(min(5, len(lstm_results)))])
        plt.ylabel('Precisión de Validación (%)')
    
    if not mlp_results.empty:
        plt.subplot(1, 2, 2)
        plt.title('Resultados MLP')
        plt.bar(range(min(5, len(mlp_results))), mlp_results['val_accuracy'].head(5))
        plt.xticks(range(min(5, len(mlp_results))), [f"Config {i+1}" for i in range(min(5, len(mlp_results)))])
        plt.ylabel('Precisión de Validación (%)')
    
    plt.tight_layout()
    plt.savefig('grid_search_results.png')
    plt.close()
    
    logger.info("Gráfico de resultados de grid search guardado como grid_search_results.png")
    
    return results_df

def load_best_model(model_path, device=None):
    """
    Cargar el mejor modelo guardado
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Cargar checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Crear modelo basado en tipo guardado
    if checkpoint['model_type'].lower() == 'lstm':
        model = VideoClassificationLSTM(
            input_size=checkpoint['input_size'],
            hidden_size=checkpoint['hidden_size'],
            num_layers=checkpoint['num_layers'],
            num_classes=2,
            dropout_rate=0.3,
            bidirectional=checkpoint['bidirectional']
        ).to(device)
    else:
        model = VideoClassificationMLP(
            input_size=checkpoint['input_size'],
            hidden_layers=checkpoint['hidden_layers'],
            num_classes=2,
            dropout_rate=0.3
        ).to(device)
    
    # Cargar estado del modelo
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, checkpoint

def predict_video(model, video_path, selected_features=None, max_frames=None, device=None):
    """
    Predecir la clase de un nuevo video
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Poner modelo en modo evaluación
    model.eval()
    
    # Cargar video como si fuera parte del dataset
    if not os.path.exists(video_path):
        raise ValueError(f"El archivo {video_path} no existe")
    
    try:
        # Leer CSV
        df = pd.read_csv(video_path)
        
        # Selección de características
        if selected_features is None:
            features = df.select_dtypes(include=[np.number])
        else:
            valid_features = [f for f in selected_features if f in df.columns]
            if not valid_features:
                raise ValueError(f"No se encontraron características válidas en {video_path}")
            features = df[valid_features]
        
        # Convertir a numpy array
        features_array = features.values
        
        # Manejar valores NaN
        if np.isnan(features_array).any():
            features_array = np.nan_to_num(features_array)
        
        # Limitar frames
        if max_frames is not None and len(features_array) > max_frames:
            features_array = features_array[:max_frames]
        
        # Normalizar
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(features_array)
        
        # Convertir a tensor
        video_tensor = torch.FloatTensor(normalized_features).unsqueeze(0)  # Añadir dimensión de lote
        
        # Mover a dispositivo
        video_tensor = video_tensor.to(device)
        
        # Realizar predicción
        with torch.no_grad():
            # Determinar tipo de modelo por su clase
            if isinstance(model, VideoClassificationLSTM):
                outputs = model(video_tensor)
            else:
                outputs = model(video_tensor)
            
            # Obtener probabilidades
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # Obtener clase predicha
            _, predicted_class = torch.max(outputs, 1)
            
            # Mapear a etiquetas legibles
            class_names = ['normal', 'sospechoso']
            predicted_label = class_names[predicted_class.item()]
            confidence = probabilities[0][predicted_class.item()].item() * 100
        
        return {
            'predicted_class': predicted_class.item(),
            'predicted_label': predicted_label,
            'confidence': confidence,
            'probabilities': {
                'normal': probabilities[0][0].item() * 100,
                'sospechoso': probabilities[0][1].item() * 100
            }
        }
    
    except Exception as e:
        logger.error(f"Error al procesar el video {video_path}: {str(e)}")
        raise

def ensemble_prediction(models, video_path, selected_features=None, max_frames=None, device=None):
    """
    Realizar predicción con conjunto de modelos
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Obtener predicciones individuales
    predictions = []
    for model in models:
        try:
            pred = predict_video(model, video_path, selected_features, max_frames, device)
            predictions.append(pred)
        except Exception as e:
            logger.error(f"Error en predicción de modelo: {str(e)}")
    
    if not predictions:
        raise ValueError("Ningún modelo pudo realizar predicciones")
    
    # Votar por la clase más común
    votes = [p['predicted_class'] for p in predictions]
    predicted_class = max(set(votes), key=votes.count)
    
    # Calcular confianza promedio para la clase predicha
    confidences = [p['probabilities']['normal'] if predicted_class == 0 else p['probabilities']['sospechoso'] 
                  for p in predictions]
    avg_confidence = sum(confidences) / len(confidences)
    
    # Mapear a etiqueta legible
    class_names = ['normal', 'sospechoso']
    predicted_label = class_names[predicted_class]
    
    return {
        'predicted_class': predicted_class,
        'predicted_label': predicted_label,
        'confidence': avg_confidence,
        'individual_predictions': predictions
    }

if __name__ == '__main__':
    # Directorio con archivos CSV de videos
    csv_directory = 'D:\Documentos\Monitoreo-Comportamientos-Sospechosos\datasetCSV'
    
    # Verificar que el directorio existe
    if not os.path.exists(csv_directory):
        logger.error(f"El directorio {csv_directory} no existe")
        exit(1)
    
    # python script.py --mode train --model_type mlp
    import argparse
    parser = argparse.ArgumentParser(description='Entrenamiento y evaluación de clasificador de videos')
    parser.add_argument('--mode', type=str, choices=['train', 'grid_search', 'predict'], default='train',
                       help='Modo de operación')
    parser.add_argument('--model_type', type=str, choices=['lstm', 'mlp'], default='lstm',
                       help='Tipo de modelo para entrenamiento')
    parser.add_argument('--video_path', type=str, help='Ruta al archivo CSV para predicción')
    parser.add_argument('--model_path', type=str, help='Ruta al modelo guardado para predicción')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        logger.info(f"Entrenando modelo {args.model_type}...")
        
        if args.model_type == 'lstm':
            train_video_classifier(
                csv_dir=csv_directory,
                model_type='lstm',
                hidden_size=128,
                num_layers=2,
                bidirectional=True,
                learning_rate=0.001,
                batch_size=32,
                dropout_rate=0.3,
                epochs=100,
                patience=15,
                max_frames=10000
            )
        else:
            train_video_classifier(
                csv_dir=csv_directory,
                model_type='mlp',
                hidden_layers=[128, 64, 32],
                learning_rate=0.001,
                batch_size=32,
                dropout_rate=0.3,
                epochs=100,
                patience=15,
                max_frames=10000
            )
    
    elif args.mode == 'grid_search':
        logger.info("Realizando grid search...")
        grid_search(csv_directory)
    
    elif args.mode == 'predict':
        if not args.video_path:
            logger.error("Se requiere la ruta del video para predicción")
            exit(1)
        
        if not args.model_path:
            logger.error("Se requiere la ruta del modelo para predicción")
            exit(1)
        
        if not os.path.exists(args.video_path):
            logger.error(f"El archivo {args.video_path} no existe")
            exit(1)
        
        if not os.path.exists(args.model_path):
            logger.error(f"El modelo {args.model_path} no existe")
            exit(1)
        
        logger.info(f"Prediciendo clase para {args.video_path}...")
        model, _ = load_best_model(args.model_path)
        result = predict_video(model, args.video_path)
        
        logger.info(f"Predicción: {result['predicted_label']}")
        logger.info(f"Confianza: {result['confidence']:.2f}%")
        logger.info(f"Probabilidades: Normal={result['probabilities']['normal']:.2f}%, Sospechoso={result['probabilities']['sospechoso']:.2f}%")