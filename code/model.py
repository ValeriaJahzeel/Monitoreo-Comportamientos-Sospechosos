import os
import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report, confusion_matrix
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
    def __init__(self, csv_dir, selected_features=None, max_frames=None, normalize=True,
                 cache_data=True, class_names=None):
        """
        Conjunto de datos para clasificación de videos con longitudes variables

        Args:
            csv_dir (str): Directorio con archivos CSV de videos
            selected_features (list, optional): Lista de características a usar
            max_frames (int, optional): Número máximo de frames a considerar
            normalize (bool): Si se debe normalizar los datos
            cache_data (bool): Si se debe almacenar en caché los datos para acceso más rápido
            class_names (list[str], optional): Nombres de clase, en el orden en que se
                asignan sus índices (0, 1, 2, ...). Cada CSV debe llamarse
                "<clase>_<nombre>.csv" (p. ej. "normal_3.csv", "merodeo_12.csv").
                Por defecto: ['normal', 'merodeo', 'forcejeo'], las tres categorías
                de comportamiento del proyecto.
        """
        self.videos = []
        self.labels = []
        self.video_lengths = []
        self.cache_data = cache_data
        self.data_cache = {}
        self.class_names = class_names if class_names is not None else ['normal', 'merodeo', 'forcejeo']
        label_map = {name: idx for idx, name in enumerate(self.class_names)}

        logger.info(f"Cargando datos de {csv_dir}...")
        start_time = time.time()

        # Verificar que el directorio existe
        if not os.path.exists(csv_dir):
            raise ValueError(f"El directorio {csv_dir} no existe")

        csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
        if len(csv_files) == 0:
            raise ValueError(f"No se encontraron archivos CSV en {csv_dir}")

        # Conteo de videos por clase (para el log de resumen)
        class_counts = {name: 0 for name in self.class_names}

        # Mapeo de etiquetas según el prefijo del nombre de archivo
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

                # Mapeo de etiquetas basado en el prefijo del nombre del archivo
                # (todo lo que va antes del primer "_")
                prefijo = filename.split('_', 1)[0]
                if prefijo not in label_map:
                    logger.warning(
                        f"Archivo {filename} ignorado - prefijo '{prefijo}' no está en "
                        f"class_names {self.class_names}"
                    )
                    continue
                label = label_map[prefijo]
                class_counts[prefijo] += 1

                # Guardar SIN normalizar todavía: normalizar aquí, por video,
                # borraría la diferencia de escala de movimiento entre videos
                # "tranquilos" y "erráticos", que es justo la señal que se
                # quiere clasificar. La normalización real se hace después con
                # fit_scaler(), ajustando un único StandardScaler global sobre
                # el split de entrenamiento y aplicándolo a todos los videos.
                video_tensor = torch.FloatTensor(features_array)

                self.videos.append(video_tensor)
                self.labels.append(label)
                self.video_lengths.append(len(video_tensor))

        # Verificar que se hayan cargado videos
        if not self.videos:
            raise ValueError("No se encontraron videos válidos. Verifica tus archivos CSV.")

        elapsed_time = time.time() - start_time
        logger.info(f"Total de videos cargados: {len(self.videos)} en {elapsed_time:.2f} segundos")
        for name, count in class_counts.items():
            logger.info(f"Videos '{name}': {count}")
        logger.info(f"Longitudes de videos - Min: {min(self.video_lengths)}, Max: {max(self.video_lengths)}, Promedio: {np.mean(self.video_lengths):.2f}")

        # Calcular dimensiones para realizar validaciones
        if len(self.videos) > 0:
            self.input_dim = self.videos[0].shape[1]
            logger.info(f"Dimensión de características: {self.input_dim}")

        self.normalize = normalize
        self.scaler = None

    def fit_scaler(self, train_indices):
        """
        Ajusta un StandardScaler global usando solo los videos en
        `train_indices` y lo aplica a TODOS los videos del dataset
        (entrenamiento y validación), reemplazando self.videos en el lugar.

        Debe llamarse una sola vez, después de dividir en train/val y antes
        de empezar a entrenar. El scaler ajustado queda en self.scaler para
        poder guardarlo junto con el modelo y reutilizarlo en inferencia.
        """
        if not self.normalize:
            return

        train_features = np.concatenate(
            [self.videos[i].numpy() for i in train_indices], axis=0
        )

        self.scaler = StandardScaler()
        self.scaler.fit(train_features)

        for i in range(len(self.videos)):
            normalized = self.scaler.transform(self.videos[i].numpy())
            self.videos[i] = torch.FloatTensor(normalized)

        # Los tensores cambiaron: cualquier caché previa quedó obsoleta
        self.data_cache = {}

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
            output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True, total_length=seq_len)
        else:
            output, (hidden, _) = self.lstm(x)

        # Mecanismo de atención
        attention_logits = self.attention(output).squeeze(-1)

        if lengths is not None:
            # Los videos más cortos que el máximo del batch quedan rellenos
            # con ceros a partir de su longitud real: sin esta máscara, el
            # softmax de atención podría repartir peso hacia esas posiciones
            # de relleno en vez de ignorarlas.
            posiciones = torch.arange(seq_len, device=output.device).unsqueeze(0)
            mascara_padding = posiciones >= lengths.to(output.device).unsqueeze(1)
            attention_logits = attention_logits.masked_fill(mascara_padding, float('-inf'))

        attention_weights = torch.softmax(attention_logits, dim=1)
        context = torch.bmm(attention_weights.unsqueeze(1), output).squeeze(1)

        # Clasificación
        logits = self.fc(context)
        
        return logits

class VideoClassificationMLP(nn.Module):
    def __init__(self, input_size, hidden_layers, num_classes, dropout_rate=0.3):
        super(VideoClassificationMLP, self).__init__()

        # Se concatenan media y máximo por característica a lo largo del
        # tiempo: el máximo conserva picos puntuales (p. ej. un pico de
        # velocidad o aceleración) que un promedio por sí solo diluye.
        pooled_size = input_size * 2

        layers = []
        prev_size = pooled_size

        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))  # Normalización por lotes
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, num_classes))

        self.model = nn.Sequential(*layers)

    def forward(self, x, lengths=None):
        # x: (batch, seq_len, features). Los videos más cortos que el máximo
        # del batch están rellenos con ceros a partir de su longitud real:
        # sin máscara, ese relleno se metería en el promedio y en el máximo.
        batch_size, seq_len, _ = x.size()

        if lengths is not None:
            posiciones = torch.arange(seq_len, device=x.device).unsqueeze(0)
            mascara = (posiciones < lengths.to(x.device).unsqueeze(1)).unsqueeze(-1).float()

            suma = (x * mascara).sum(dim=1)
            conteo = mascara.sum(dim=1).clamp(min=1)
            mean_pool = suma / conteo

            x_para_max = x.masked_fill(mascara == 0, float('-inf'))
            max_pool, _ = x_para_max.max(dim=1)
            max_pool = torch.where(torch.isinf(max_pool), torch.zeros_like(max_pool), max_pool)
        else:
            mean_pool = x.mean(dim=1)
            max_pool, _ = x.max(dim=1)

        pooled = torch.cat([mean_pool, max_pool], dim=1)

        return self.model(pooled)

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
                           class_names=None,   # ['normal', 'merodeo', 'forcejeo'] por defecto
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
    dataset = VideoFrameDataset(csv_dir, selected_features, max_frames, class_names=class_names)
    num_classes = len(dataset.class_names)

    # Verificar si hay suficientes muestras para la estratificación
    class_sample_counts = [dataset.labels.count(i) for i in range(num_classes)]
    min_samples_per_class = min(class_sample_counts)
    if min_samples_per_class < 2:
        counts_str = ", ".join(f"{name}={c}" for name, c in zip(dataset.class_names, class_sample_counts))
        logger.warning(f"Pocas muestras para estratificación. Clases: {counts_str}")
        stratify = None
    else:
        stratify = dataset.labels

    # Dividir datos. Con datasets muy pequeños, incluso con >=2 muestras por
    # clase, el tamaño del split de test puede terminar siendo menor que el
    # número de clases (sklearn no puede estratificar en ese caso) — se cae
    # a una división simple sin estratificar.
    try:
        train_indices, val_indices = train_test_split(
            range(len(dataset)), test_size=0.2, stratify=stratify, random_state=42
        )
    except ValueError as e:
        logger.warning(f"No se pudo estratificar el split ({e}). Usando división simple.")
        train_indices, val_indices = train_test_split(
            range(len(dataset)), test_size=0.2, stratify=None, random_state=42
        )

    # Ajustar la normalización SOLO con el split de entrenamiento y aplicarla
    # a todo el dataset (evita fugas de información del set de validación y
    # evita que cada video se normalice con su propia media/desviación)
    dataset.fit_scaler(train_indices)

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
        # True: el MLP necesita las longitudes reales para no promediar/
        # tomar el máximo sobre el padding (ver VideoClassificationMLP.forward)
        use_lengths = True
    
    # Contar y loggear parámetros del modelo
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total de parámetros: {total_params:,}")
    logger.info(f"Parámetros entrenables: {trainable_params:,}")
    
    print_gpu_memory("Después de crear el modelo")
    
    # Determinar pesos para clases desbalanceadas
    if len(set(class_sample_counts)) > 1:
        weights = torch.FloatTensor(
            [len(dataset) / (num_classes * count) for count in class_sample_counts]
        ).to(device)
        logger.info(f"Usando pesos para clases desbalanceadas: {weights}")
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Optimizador con decaimiento de pesos para regularización
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Scheduler para reducir la tasa de aprendizaje cuando la pérdida se estanca
    # verbose se quitó de ReduceLROnPlateau en versiones recientes de PyTorch
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Variables para early stopping. -1 (no 0) asegura que se guarde un
    # checkpoint incluso si la primera época obtiene 0.00% de val_accuracy
    # (posible con validaciones muy pequeñas): si no, "> best_val_accuracy"
    # nunca sería cierto y nunca se guardaría ningún modelo.
    best_val_accuracy = -1
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
                'class_names': dataset.class_names,
                'hidden_size': hidden_size if model_type.lower() == 'lstm' else None,
                'num_layers': num_layers if model_type.lower() == 'lstm' else None,
                'hidden_layers': hidden_layers if model_type.lower() == 'mlp' else None,
                'bidirectional': bidirectional if model_type.lower() == 'lstm' else None,
                # Se guarda la media/desviación del scaler ajustado en
                # entrenamiento para poder normalizar los videos nuevos de la
                # misma forma en inferencia (ver load_scaler_from_checkpoint)
                'scaler_mean': dataset.scaler.mean_ if dataset.scaler is not None else None,
                'scaler_scale': dataset.scaler.scale_ if dataset.scaler is not None else None,
            }, model_filename)
            
            logger.info(f"Modelo guardado en {model_filename}")
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            logger.info(f"No hay mejora en el rendimiento. Counter: {early_stopping_counter}/{patience}")
        
        # Early stopping
        if early_stopping_counter >= patience:
            logger.info(f"Early stopping después de {epoch+1} épocas")
            break
    
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
    model_filename = f"best_model_{model_type}.pth"
    if os.path.exists(model_filename):
        # weights_only=False: el checkpoint incluye arrays de numpy (el
        # scaler) además de los pesos, y es un archivo generado por nosotros
        # mismos en el paso anterior de este mismo entrenamiento (fuente de
        # confianza).
        checkpoint = torch.load(model_filename, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Evaluación final
        _, final_val_accuracy, _, _ = validate(model, val_loader, criterion, device, use_lengths)
        logger.info(f"Precisión final del mejor modelo: {final_val_accuracy:.2f}%")
    else:
        logger.warning(f"No se guardó ningún checkpoint en {model_filename}; se omite la evaluación final.")

    return best_val_accuracy, history

def cross_validar(csv_dir,
                  model_type='lstm',
                  selected_features=None,
                  max_frames=None,
                  class_names=None,
                  n_splits=5,
                  hidden_size=64,
                  num_layers=2,
                  hidden_layers=[64, 32],
                  learning_rate=0.001,
                  epochs=100,
                  batch_size=16,
                  dropout_rate=0.3,
                  bidirectional=True,
                  patience=10,
                  weight_decay=1e-5,
                  random_state=42):
    """
    Evalúa LSTM/MLP con validación cruzada estratificada de n_splits sobre
    TODO el dataset, en vez de un único split de entrenamiento/validación.

    Con datasets chicos (decenas de videos), un solo split deja la métrica
    de validación con muy pocas muestras (p. ej. 16 de 78) y demasiada
    varianza para ser un número confiable — cada video vale varios puntos
    porcentuales de accuracy. Esta función usa el mismo criterio que ya se
    aplica a los modelos de árboles en modelo_arboles.py, para poder
    comparar los 4 modelos con el mismo estándar.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"[CV {model_type}] Utilizando dispositivo: {device}")

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    dataset = VideoFrameDataset(csv_dir, selected_features, max_frames, class_names=class_names)
    num_classes = len(dataset.class_names)
    input_size = dataset.videos[0].shape[1]
    labels_array = np.array(dataset.labels)

    class_sample_counts = [dataset.labels.count(i) for i in range(num_classes)]
    n_splits_efectivo = min(n_splits, min(class_sample_counts))
    if n_splits_efectivo < 2:
        logger.warning("No hay suficientes muestras por clase para validación cruzada.")
        return None

    skf = StratifiedKFold(n_splits=n_splits_efectivo, shuffle=True, random_state=random_state)
    num_workers = min(4, os.cpu_count() or 1)

    fold_accuracies = []
    fold_f1s = []
    y_true_oof = []
    y_pred_oof = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels_array)), labels_array)):
        logger.info(f"[CV {model_type}] Fold {fold_idx + 1}/{n_splits_efectivo}")

        # Normalizar SOLO con el fold de entrenamiento de esta iteración
        # (misma razón que en train_video_classifier: evita que el scaler
        # vea datos del fold de validación)
        dataset.fit_scaler(train_idx)

        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)

        train_loader = DataLoader(
            train_subset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
            num_workers=num_workers, pin_memory=torch.cuda.is_available()
        )
        val_loader = DataLoader(
            val_subset, batch_size=batch_size, collate_fn=collate_fn,
            num_workers=num_workers, pin_memory=torch.cuda.is_available()
        )

        if model_type.lower() == 'lstm':
            model = VideoClassificationLSTM(
                input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                num_classes=num_classes, dropout_rate=dropout_rate, bidirectional=bidirectional
            ).to(device)
        else:
            model = VideoClassificationMLP(
                input_size=input_size, hidden_layers=hidden_layers,
                num_classes=num_classes, dropout_rate=dropout_rate
            ).to(device)
        use_lengths = True

        fold_counts = [int((labels_array[train_idx] == i).sum()) for i in range(num_classes)]
        if len(set(fold_counts)) > 1 and min(fold_counts) > 0:
            weights = torch.FloatTensor(
                [len(train_idx) / (num_classes * c) for c in fold_counts]
            ).to(device)
            criterion = nn.CrossEntropyLoss(weight=weights)
        else:
            criterion = nn.CrossEntropyLoss()

        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        best_val_accuracy = -1
        best_preds, best_labels = None, None
        early_stopping_counter = 0

        for epoch in range(epochs):
            train_loss, train_accuracy = train_one_epoch(
                model, train_loader, criterion, optimizer, device, use_lengths
            )
            val_loss, val_accuracy, val_preds, val_labels = validate(
                model, val_loader, criterion, device, use_lengths
            )
            scheduler.step(val_loss)

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_preds, best_labels = val_preds, val_labels
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            if early_stopping_counter >= patience:
                break

        logger.info(f"[CV {model_type}] Fold {fold_idx + 1}: mejor val accuracy = {best_val_accuracy:.2f}% "
                    f"({epoch + 1} épocas)")

        fold_accuracies.append(best_val_accuracy / 100.0)
        fold_f1s.append(f1_score(best_labels, best_preds, average='weighted', zero_division=0))
        y_true_oof.extend(best_labels)
        y_pred_oof.extend(best_preds)

        # Liberar memoria de GPU entre folds
        del model, optimizer, scheduler
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    acc_mean, acc_std = float(np.mean(fold_accuracies)), float(np.std(fold_accuracies))
    f1_mean, f1_std = float(np.mean(fold_f1s)), float(np.std(fold_f1s))

    etiquetas_posibles = list(range(num_classes))

    logger.info(f"\n=== {model_type.upper()} - Validación cruzada {n_splits_efectivo}-fold ===")
    logger.info(f"Accuracy: {acc_mean*100:.1f}% ± {acc_std*100:.1f}%  |  "
                f"F1 (weighted): {f1_mean:.3f} ± {f1_std:.3f}")
    logger.info(f"Accuracy por fold: {[f'{a*100:.1f}%' for a in fold_accuracies]}")
    logger.info("\n" + classification_report(
        y_true_oof, y_pred_oof, labels=etiquetas_posibles,
        target_names=dataset.class_names, zero_division=0
    ))
    logger.info("Matriz de confusión out-of-fold (filas=real, columnas=predicho):")
    logger.info("\n" + str(confusion_matrix(y_true_oof, y_pred_oof, labels=etiquetas_posibles)))

    return {'accuracy_mean': acc_mean, 'accuracy_std': acc_std, 'f1_mean': f1_mean, 'f1_std': f1_std}

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
    
    # Cargar checkpoint. weights_only=False: incluye arrays de numpy (el
    # scaler) además de los pesos; se asume que model_path es un checkpoint
    # propio (entrenado con este mismo código), no un archivo de origen
    # externo/no confiable.
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Modelos guardados antes de soportar múltiples clases no incluyen
    # class_names: se asume el par binario original como respaldo
    class_names = checkpoint.get('class_names', ['normal', 'sospechoso'])
    num_classes = len(class_names)

    # Crear modelo basado en tipo guardado
    if checkpoint['model_type'].lower() == 'lstm':
        model = VideoClassificationLSTM(
            input_size=checkpoint['input_size'],
            hidden_size=checkpoint['hidden_size'],
            num_layers=checkpoint['num_layers'],
            num_classes=num_classes,
            dropout_rate=0.3,
            bidirectional=checkpoint['bidirectional']
        ).to(device)
    else:
        model = VideoClassificationMLP(
            input_size=checkpoint['input_size'],
            hidden_layers=checkpoint['hidden_layers'],
            num_classes=num_classes,
            dropout_rate=0.3
        ).to(device)
    
    # Cargar estado del modelo
    model.load_state_dict(checkpoint['model_state_dict'])

    return model, checkpoint

def load_scaler_from_checkpoint(checkpoint):
    """
    Reconstruye el StandardScaler ajustado durante el entrenamiento a partir
    de la media/desviación guardadas en el checkpoint. Devuelve None si el
    checkpoint no incluye esa información (por ejemplo, modelos guardados
    antes de este cambio).
    """
    mean = checkpoint.get('scaler_mean')
    scale = checkpoint.get('scaler_scale')
    if mean is None or scale is None:
        return None

    scaler = StandardScaler()
    scaler.mean_ = mean
    scaler.scale_ = scale
    scaler.var_ = scale ** 2
    scaler.n_features_in_ = len(mean)
    return scaler

def predict_video(model, video_path, selected_features=None, max_frames=None, device=None,
                   scaler=None, class_names=None):
    """
    Predecir la clase de un nuevo video.

    `scaler` debe ser el StandardScaler ajustado durante el entrenamiento
    (ver load_scaler_from_checkpoint). Si no se provee, se usa un scaler
    ajustado únicamente con este video como método de respaldo, pero eso
    normaliza el video de forma distinta a como se entrenó el modelo y puede
    dar predicciones poco confiables.

    `class_names` debe ser la misma lista con la que se entrenó el modelo
    (ver checkpoint['class_names']). Por defecto ['normal', 'merodeo',
    'forcejeo'].
    """
    if class_names is None:
        class_names = ['normal', 'merodeo', 'forcejeo']
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
        
        # Normalizar con el scaler del entrenamiento si está disponible;
        # si no, ajustar uno solo con este video como respaldo
        if scaler is not None:
            normalized_features = scaler.transform(features_array)
        else:
            logger.warning(
                "predict_video sin scaler de entrenamiento: normalizando solo "
                "con este video, la predicción puede ser poco confiable"
            )
            fallback_scaler = StandardScaler()
            normalized_features = fallback_scaler.fit_transform(features_array)

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
            predicted_label = class_names[predicted_class.item()]
            confidence = probabilities[0][predicted_class.item()].item() * 100

        return {
            'predicted_class': predicted_class.item(),
            'predicted_label': predicted_label,
            'confidence': confidence,
            'probabilities': {
                name: probabilities[0][i].item() * 100
                for i, name in enumerate(class_names)
            }
        }
    
    except Exception as e:
        logger.error(f"Error al procesar el video {video_path}: {str(e)}")
        raise

def ensemble_prediction(model_checkpoints, video_path, selected_features=None, max_frames=None, device=None):
    """
    Realizar predicción con conjunto de modelos.

    `model_checkpoints` es una lista de tuplas (model, checkpoint), donde
    `checkpoint` es el diccionario devuelto por load_best_model para ese
    modelo (se usa para recuperar el scaler con el que se entrenó cada uno).
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not model_checkpoints:
        raise ValueError("Se requiere al menos un (modelo, checkpoint) para el ensemble")

    # Se asume que todos los modelos del ensemble comparten las mismas clases
    # (de lo contrario no tiene sentido promediar/votar entre ellos)
    class_names = model_checkpoints[0][1].get('class_names', ['normal', 'merodeo', 'forcejeo'])

    # Obtener predicciones individuales
    predictions = []
    for model, checkpoint in model_checkpoints:
        try:
            scaler = load_scaler_from_checkpoint(checkpoint)
            model_class_names = checkpoint.get('class_names', class_names)
            pred = predict_video(model, video_path, selected_features, max_frames, device, scaler, model_class_names)
            predictions.append(pred)
        except Exception as e:
            logger.error(f"Error en predicción de modelo: {str(e)}")

    if not predictions:
        raise ValueError("Ningún modelo pudo realizar predicciones")

    # Votar por la clase más común
    votes = [p['predicted_class'] for p in predictions]
    predicted_class = max(set(votes), key=votes.count)

    # Calcular confianza promedio para la clase predicha
    predicted_label = class_names[predicted_class]
    confidences = [p['probabilities'][predicted_label] for p in predictions]
    avg_confidence = sum(confidences) / len(confidences)
    
    return {
        'predicted_class': predicted_class,
        'predicted_label': predicted_label,
        'confidence': avg_confidence,
        'individual_predictions': predictions
    }

if __name__ == '__main__':
    # Directorio por defecto con archivos CSV de videos: la salida directa de
    # objectDetection.py (relativo a la raíz del proyecto; este script vive
    # en code/)
    default_csv_directory = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'informacion', 'csv'
    )

    # python script.py --mode train --model_type mlp
    import argparse
    parser = argparse.ArgumentParser(description='Entrenamiento y evaluación de clasificador de videos')
    parser.add_argument('--mode', type=str, choices=['train', 'grid_search', 'predict', 'cross_validate'], default='train',
                       help='Modo de operación')
    parser.add_argument('--n_splits', type=int, default=5, help='Número de folds para --mode cross_validate')
    parser.add_argument('--model_type', type=str, choices=['lstm', 'mlp'], default='lstm',
                       help='Tipo de modelo para entrenamiento')
    parser.add_argument('--csv_dir', type=str, default=default_csv_directory,
                       help='Directorio con archivos CSV de videos para entrenamiento/grid search')
    parser.add_argument('--video_path', type=str, help='Ruta al archivo CSV para predicción')
    parser.add_argument('--model_path', type=str, help='Ruta al modelo guardado para predicción')

    args = parser.parse_args()
    csv_directory = args.csv_dir

    if args.mode in ('train', 'grid_search', 'cross_validate') and not os.path.exists(csv_directory):
        logger.error(f"El directorio {csv_directory} no existe")
        exit(1)

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
                epochs=50,
                patience=15,
                max_frames=10000
            )
        else:
            # train_video_classifier(
            #     csv_dir=csv_directory,
            #     model_type='mlp',
            #     hidden_layers=[256, 128, 64],  # Arquitectura modificada
            #     learning_rate=0.0005,  # Tasa de aprendizaje modificada
            #     batch_size=16,  # Tamaño de lote modificado
            #     dropout_rate=0.4,  # Dropout modificado
            #     epochs=150,  # Más épocas
            #     patience=20,  # Más paciencia
            #     weight_decay=1e-4  # Regularización modificada
            # )
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

    elif args.mode == 'cross_validate':
        logger.info(f"Validación cruzada ({args.n_splits}-fold) para {args.model_type}...")
        # Mismos hiperparámetros que --mode train, para que la comparación
        # entre split único y validación cruzada sea directa
        if args.model_type == 'lstm':
            cross_validar(
                csv_dir=csv_directory,
                model_type='lstm',
                n_splits=args.n_splits,
                hidden_size=128,
                num_layers=2,
                bidirectional=True,
                learning_rate=0.001,
                batch_size=32,
                dropout_rate=0.3,
                epochs=50,
                patience=15,
                max_frames=10000
            )
        else:
            cross_validar(
                csv_dir=csv_directory,
                model_type='mlp',
                n_splits=args.n_splits,
                hidden_layers=[128, 64, 32],
                learning_rate=0.001,
                batch_size=32,
                dropout_rate=0.3,
                epochs=100,
                patience=15,
                max_frames=10000
            )
    
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
        model, checkpoint = load_best_model(args.model_path)
        scaler = load_scaler_from_checkpoint(checkpoint)
        class_names = checkpoint.get('class_names', ['normal', 'merodeo', 'forcejeo'])
        result = predict_video(model, args.video_path, scaler=scaler, class_names=class_names)
        
        probs_str = ", ".join(f"{name}={p:.2f}%" for name, p in result['probabilities'].items())
        logger.info(f"Predicción: {result['predicted_label']}")
        logger.info(f"Confianza: {result['confidence']:.2f}%")
        logger.info(f"Probabilidades: {probs_str}")