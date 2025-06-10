# lstm_v2_improved.py
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Masking, LSTM, BatchNormalization, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import uniform_filter1d
import warnings
warnings.filterwarnings('ignore')

# ---------- Configuración mejorada ----------
plt.style.use('seaborn-v0_8-darkgrid')
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)

# ---------- Dataset secuencial mejorado ----------
class VideoDataset:
    def __init__(self, max_len=50):
        self.max_len = max_len
        self.scaler = StandardScaler()
        self.enc = LabelEncoder()
        # Features expandidas
        self.cols = ['Velocidad', 'Aceleracion', 'Linealidad',
                     'Circularidad', 'Zigzag', 'Densidad', 'Area_Trayectoria',
                     'Centroide_X', 'Centroide_Y']
        
    def _compute_additional_features(self, df):
        """Calcula features adicionales avanzadas"""
        features = []
        n = len(df)
        
        # Velocidad angular
        if n > 1:
            dx = df['Centroide_X'].diff()
            dy = df['Centroide_Y'].diff()
            angles = np.arctan2(dy, dx)
            angular_vel = np.abs(angles.diff())
            features.append(angular_vel.fillna(0).values)
        else:
            features.append(np.zeros(n))
            
        # Jerk (derivada de aceleración)
        if 'Aceleracion' in df.columns and n > 1:
            jerk = df['Aceleracion'].diff().fillna(0)
            features.append(jerk.values)
        else:
            features.append(np.zeros(n))
            
        # Entropía de movimiento (medida de aleatoriedad)
        if n > 5:
            vel_hist, _ = np.histogram(df['Velocidad'].values, bins=10)
            vel_hist = vel_hist + 1e-10  # Evitar log(0)
            entropy = -np.sum(vel_hist * np.log(vel_hist/vel_hist.sum()))
            features.append(np.full(n, entropy))
        else:
            features.append(np.zeros(n))
            
        return np.column_stack(features) if features else np.zeros((n, 0))

    def _csv2seq(self, csv):
        try:
            df = pd.read_csv(csv)
            seqs = []
            label = ('normal' if 'normal' in csv.stem else
                     'merodeo' if 'merodeo' in csv.stem else 'forcejeo')
            
            for _, traj in df.groupby('Objeto'):
                traj = traj.sort_values('Frame')
                
                # Features originales
                x_base = traj[self.cols].values.astype('float32')
                
                # Features adicionales
                x_add = self._compute_additional_features(traj)
                
                # Concatenar todas las features
                x = np.hstack([x_base, x_add]) if x_add.size > 0 else x_base
                
                # Truncar o hacer padding para asegurar longitud max_len
                if len(x) > self.max_len:
                    x = x[:self.max_len]  # Truncar al máximo
                elif len(x) < self.max_len:
                    padding = np.zeros((self.max_len-len(x), x.shape[1]), dtype='float32')
                    x = np.vstack([x, padding])  # Padding con ceros
                    
                seqs.append((x, label, csv.stem))
            return seqs
        except Exception as e:
            print(f"Error processing {csv}: {str(e)}")
            return []

    def load(self, root):
        X, y, g = [], [], []
        for cls in ['normal', 'merodeo', 'forcejeo']:
            cls_path = Path(root) / cls
            if not cls_path.exists():
                print(f"Warning: Directory {cls_path} does not exist")
                continue
                
            for csv in cls_path.glob('*.csv'):
                seq_data = self._csv2seq(csv)
                for s, l, v in seq_data:
                    X.append(s)
                    y.append(l)
                    g.append(v)
        
        if not X:
            raise ValueError("No data loaded. Check your data paths and contents.")
            
        self.X = np.stack(X)
        self.y = self.enc.fit_transform(y)
        self.g = np.array(g)
        print(f"Secuencias: {self.X.shape} | Vídeos únicos: {len(set(g))}")
        print(f"Clases: {dict(zip(self.enc.classes_, np.bincount(self.y)))}")

    def scale(self, tr_idx, te_idx):
        n, t, f = self.X.shape
        Xt, Xv = self.X[tr_idx], self.X[te_idx]
        
        # Reshape para escalado
        Xt_2d = Xt.reshape(-1, f)
        self.scaler.fit(Xt_2d)
        
        # Transformar y volver a la forma original
        Xt_scaled = self.scaler.transform(Xt_2d).reshape(len(tr_idx), t, f)
        Xv_scaled = self.scaler.transform(Xv.reshape(-1, f)).reshape(len(te_idx), t, f)
        
        return Xt_scaled, Xv_scaled

# ---------- Data Augmentation mejorada ----------
def augment_batch(x, noise_std=0.01):
    """Augmentation avanzada con múltiples técnicas"""
    x_aug = x.copy()
    batch_size = len(x)
    max_len = x.shape[1]
    
    # 1. Ruido gaussiano
    noise = np.random.normal(0, noise_std, x.shape).astype('float32')
    x_aug += noise
    
    # 2. Time warping (estirar/comprimir temporal)
    for i in range(batch_size):
        if np.random.rand() < 0.3:
            # Encontrar longitud real (sin padding)
            non_zero = np.any(x[i] != 0, axis=1)
            real_len = np.sum(non_zero)
            
            if real_len > 10:
                warp_factor = np.random.uniform(0.8, 1.2)
                warped_len = min(int(real_len * warp_factor), max_len)
                
                if warped_len > 0:
                    indices = np.linspace(0, real_len-1, warped_len)
                    indices = np.clip(indices.astype(int), 0, real_len-1)
                    
                    # Aplicar warping
                    temp = np.zeros_like(x[i])
                    temp[:warped_len] = x[i][indices]
                    x_aug[i] = temp
    
    # 3. Escalamiento
    if np.random.rand() < 0.3:
        scale_factor = np.random.uniform(0.8, 1.2, size=(batch_size, 1, x.shape[2]))
        x_aug *= scale_factor
    
    # 4. Reversión temporal
    if np.random.rand() < 0.2:
        mask = np.random.rand(batch_size) < 0.2
        x_aug[mask] = x_aug[mask, ::-1, :]
    
    # 5. Mixup entre muestras del mismo batch
    if np.random.rand() < 0.2 and batch_size > 1:
        lambda_mix = np.random.beta(0.2, 0.2)
        indices = np.random.permutation(batch_size)
        x_aug = lambda_mix * x_aug + (1 - lambda_mix) * x_aug[indices]
    
    return x_aug

class AugmentSeq(Sequence):
    def __init__(self, X, y, batch_size=8):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.indices = np.arange(len(X))
        
    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        x_batch = self.X[batch_indices]
        y_batch = self.y[batch_indices]
        
        return augment_batch(x_batch), y_batch
    
    def on_epoch_end(self):
        np.random.shuffle(self.indices)

# ---------- Callback para F1 Score ----------
class F1Callback(Callback):
    def __init__(self, X_val, y_val, patience=8):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.patience = patience
        self.best_f1 = 0
        self.wait = 0
        self.f1_history = []
        
    def on_epoch_end(self, epoch, logs=None):
        y_pred = np.argmax(self.model.predict(self.X_val, verbose=0), axis=1)
        f1 = f1_score(self.y_val, y_pred, average='macro')
        self.f1_history.append(f1)
        
        logs = logs or {}
        logs['val_f1'] = f1
        
        print(f' - val_f1: {f1:.4f}')
        
        if f1 > self.best_f1 + 0.001:
            self.best_f1 = f1
            self.wait = 0
            self.model.save_weights('best.weights.h5')
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print(f'\nEarly stopping triggered. Best F1: {self.best_f1:.4f}')
                self.model.stop_training = True

# ---------- Modelo LSTM mejorado ----------
def build_lstm_improved(input_shape, n_cls, units=64, drop=0.4, lr=1e-4):
    """LSTM con arquitectura mejorada"""
    model = Sequential([
        Masking(mask_value=0., input_shape=input_shape),
        
        # Primera capa LSTM con return_sequences para apilar
        LSTM(units, dropout=drop, recurrent_dropout=0.2, return_sequences=True),
        BatchNormalization(),
        
        # Segunda capa LSTM
        LSTM(units, dropout=drop, recurrent_dropout=0.2, return_sequences=True),
        BatchNormalization(),
        
        # Tercera capa LSTM
        LSTM(units//2, dropout=drop, recurrent_dropout=0.2),
        BatchNormalization(),
        
        # Capas densas
        Dense(units//2, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        Dropout(drop),
        Dense(units//4, activation='relu'),
        Dropout(drop/2),
        
        # Capa de salida
        Dense(n_cls, activation='softmax')
    ])
    
    optimizer = Adam(learning_rate=lr, clipvalue=1.0)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# ---------- Función de visualización mejorada ----------
def plot_training_history(histories, fold_scores, save_path='training_results.png'):
    """Genera gráficas completas del entrenamiento"""
    plt.figure(figsize=(20, 15))
    
    # 1. Loss promedio por época
    plt.subplot(3, 3, 1)
    all_loss = np.array([h['loss'] for h in histories])
    all_val_loss = np.array([h['val_loss'] for h in histories])
    
    mean_loss = np.mean(all_loss, axis=0)
    std_loss = np.std(all_loss, axis=0)
    mean_val_loss = np.mean(all_val_loss, axis=0)
    std_val_loss = np.std(all_val_loss, axis=0)
    
    epochs = range(1, len(mean_loss) + 1)
    plt.plot(epochs, mean_loss, 'b-', label='Train Loss', linewidth=2)
    plt.fill_between(epochs, mean_loss - std_loss, mean_loss + std_loss, color='blue', alpha=0.2)
    plt.plot(epochs, mean_val_loss, 'r-', label='Val Loss', linewidth=2)
    plt.fill_between(epochs, mean_val_loss - std_val_loss, mean_val_loss + std_val_loss, color='red', alpha=0.2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Evolution (Mean ± Std)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Accuracy promedio por época
    plt.subplot(3, 3, 2)
    all_acc = np.array([h['accuracy'] for h in histories])
    all_val_acc = np.array([h['val_accuracy'] for h in histories])
    
    mean_acc = np.mean(all_acc, axis=0)
    std_acc = np.std(all_acc, axis=0)
    mean_val_acc = np.mean(all_val_acc, axis=0)
    std_val_acc = np.std(all_val_acc, axis=0)
    
    plt.plot(epochs, mean_acc, 'b-', label='Train Acc', linewidth=2)
    plt.fill_between(epochs, mean_acc - std_acc, mean_acc + std_acc, color='blue', alpha=0.2)
    plt.plot(epochs, mean_val_acc, 'r-', label='Val Acc', linewidth=2)
    plt.fill_between(epochs, mean_val_acc - std_val_acc, mean_val_acc + std_val_acc, color='red', alpha=0.2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Evolution (Mean ± Std)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. F1 Score por fold
    plt.subplot(3, 3, 3)
    fold_names = [f'Fold {i+1}' for i in range(len(fold_scores))]
    bars = plt.bar(fold_names, fold_scores)
    
    # Colorear barras según performance
    for bar, score in zip(bars, fold_scores):
        if score >= 0.85:
            bar.set_color('green')
        elif score >= 0.80:
            bar.set_color('orange')
        else:
            bar.set_color('red')
        bar.set_alpha(0.7)
    
    plt.axhline(y=np.mean(fold_scores), color='blue', linestyle='--', 
                label=f'Mean: {np.mean(fold_scores):.3f}')
    plt.ylabel('F1 Score')
    plt.title('F1 Score by Fold')
    plt.legend()
    plt.ylim([0, 1])
    plt.grid(True, alpha=0.3)
    
    # Añadir valores sobre las barras
    for bar, score in zip(bars, fold_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{score:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_average_confusion_matrix(matrices, class_names):
    """Plotea la matriz de confusión promedio"""
    avg_cm = np.mean(matrices, axis=0)
    avg_cm_norm = avg_cm / avg_cm.sum(axis=1, keepdims=True)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(avg_cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Proportion'})
    
    # Añadir counts en las celdas
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j + 0.5, i + 0.7, f'({int(avg_cm[i, j])})',
                     ha='center', va='center', fontsize=9, style='italic')
    
    plt.title('Average Confusion Matrix (6-Fold CV)', fontsize=14)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix_avg.png', dpi=300, bbox_inches='tight')
    plt.show()

# ---------- Cross-validation mejorado ----------
def cv_lstm_improved(ds, epochs=80, batch_size=8):
    n_cls = len(ds.enc.classes_)
    cv = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=42)
    
    scores = []
    histories = []
    confusion_matrices = []
    
    print("="*60)
    print("Starting 6-Fold Cross-Validation")
    print("="*60)
    
    for fold, (tr_idx, te_idx) in enumerate(cv.split(ds.X, ds.y, ds.g), 1):
        print(f"\nFold {fold}/6:")
        print("-"*40)
        
        # Preparar datos
        X_train, X_val = ds.scale(tr_idx, te_idx)
        y_train, y_val = ds.y[tr_idx], ds.y[te_idx]
        y_train_cat = to_categorical(y_train, n_cls)
        
        # Class weights
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weights = dict(enumerate(class_weights))
        
        # Construir modelo
        model = build_lstm_improved(X_train.shape[1:], n_cls, units=64, drop=0.3, lr=5e-4)
        
        # Callbacks
        callbacks = [
            F1Callback(X_val, y_val, patience=10),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, 
                             min_lr=1e-6, verbose=1),
            EarlyStopping(monitor='val_loss', patience=15, 
                         restore_best_weights=True, verbose=1)
        ]
        
        # Data generator con augmentation
        train_gen = AugmentSeq(X_train, y_train_cat, batch_size)
        
        # Entrenar
        history = model.fit(
            train_gen,
            epochs=epochs,
            validation_data=(X_val, to_categorical(y_val, n_cls)),
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        # Cargar mejores pesos
        model.load_weights('best_weights.h5')
        
        # Evaluar
        y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
        
        # Post-procesamiento: suavizado temporal
        if len(y_pred) > 3:
            y_pred_smooth = uniform_filter1d(y_pred.astype(float), size=3, mode='nearest')
            y_pred = np.round(y_pred_smooth).astype(int)
        
        # Métricas
        f1 = f1_score(y_val, y_pred, average='macro')
        cm = confusion_matrix(y_val, y_pred)
        
        scores.append(f1)
        histories.append(history.history)
        confusion_matrices.append(cm)
        
        print(f"\nFold {fold} Results:")
        print(f"F1 Score: {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_val, y_pred, target_names=ds.enc.classes_))
        
    # Resultados finales
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"F1-macro: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    print(f"Best fold: {np.argmax(scores)+1} (F1={np.max(scores):.4f})")
    print(f"Worst fold: {np.argmin(scores)+1} (F1={np.min(scores):.4f})")
    
    # Generar visualizaciones
    plot_training_history(histories, scores)
    plot_average_confusion_matrix(confusion_matrices, ds.enc.classes_)
    
    return scores, histories, confusion_matrices

# ---------- MAIN ----------
if __name__ == "__main__":
    # Configuración
    ROOT = r"D:\Documentos\Monitoreo-Comportamientos-Sospechosos\csv"
    
    try:
        # Cargar datos
        print("Loading dataset...")
        ds = VideoDataset(max_len=50)
        ds.load(ROOT)
        
        # Ejecutar cross-validation mejorado
        scores, histories, cms = cv_lstm_improved(ds, epochs=80, batch_size=16)
        
        # Guardar resultados
        results = {
            'scores': scores,
            'mean_f1': np.mean(scores),
            'std_f1': np.std(scores),
            'histories': histories,
            'confusion_matrices': cms,
            'class_names': ds.enc.classes_
        }
        
        import joblib
        joblib.dump(results, 'lstm_v2_improved_results.pkl')
        print("\nResults saved to 'lstm_v2_improved_results.pkl'")
        
        # Generar reporte adicional
        print("\n" + "="*60)
        print("ADDITIONAL METRICS")
        print("="*60)
        
        # Calcular métricas por clase
        avg_cm = np.mean(cms, axis=0)
        for i, class_name in enumerate(ds.enc.classes_):
            tp = avg_cm[i, i]
            fp = np.sum(avg_cm[:, i]) - tp
            fn = np.sum(avg_cm[i, :]) - tp
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"\n{class_name}:")
            print(f"  Precision: {precision:.3f}")
            print(f"  Recall: {recall:.3f}")
            print(f"  F1-Score: {f1:.3f}")
            
    except Exception as e:
        print(f"\nError during execution: {str(e)}")
        raise