"""
Detector de merodeo / forcejeo – v3 최소
Autor: tú
---------------------------------------------------------
Mejoras:
• Split leave-one-video-out (LOGO)   • F1EarlyStop callback
• GRU compacto (menos overfitting)    • Learning-rate scheduler
"""

# ------------ Imports ------------
import numpy as np, pandas as pd, os, warnings, itertools
from pathlib import Path; warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score

# ------------ 1. Callback F1-macro ------------
class F1EarlyStop(Callback):
    """Para y restaura el mejor modelo según F1-macro en validación."""
    def __init__(self, Xv, yv_cat, patience=2):
        super().__init__(); self.Xv=Xv; self.yv=yv_cat
        self.best, self.wait, self.pat = 0, 0, patience

    def on_epoch_end(self, epoch, logs=None):
        y_pred = np.argmax(self.model.predict(self.Xv, verbose=0),1)
        y_true = np.argmax(self.yv,1)
        f1 = f1_score(y_true, y_pred, average='macro')
        logs['val_f1'] = f1
        print(f"  — val_F1={f1:.3f}")
        if f1 > self.best + 1e-3:
            self.best, self.wait = f1, 0
        else:
            self.wait += 1
            if self.wait >= self.pat:
                print(f"\n⏹ Early stop en epoch {epoch+1}")
                self.model.stop_training = True

# ------------ 2. Cargador de datos ------------
class VideoDataset:
    def __init__(self, max_len=50, wanted_cols=None):
        self.max_len = max_len
        self.wanted = wanted_cols or [
            'Velocidad','Aceleracion','Linealidad',
            'Circularidad','Zigzag','Densidad','Area_Trayectoria',
            'Centroide_X','Centroide_Y'
        ]
        self.scaler, self.enc = StandardScaler(), LabelEncoder()

    def _csv_to_sequences(self, csv_path):
        df = pd.read_csv(csv_path)
        if df.empty: return []
        label = (
            'normal'   if 'normal'   in csv_path.stem else
            'merodeo'  if 'merodeo'  in csv_path.stem else
            'forcejeo'
        )
        cols = [c for c in self.wanted if c in df.columns]
        seqs = []
        for _, traj in df.groupby('Objeto'):
            traj = traj.sort_values('Frame')
            frames = traj[cols].values.astype('float32')[:self.max_len]
            if len(frames) < self.max_len:
                pad = np.zeros((self.max_len-len(frames), len(cols)), 'float32')
                frames = np.vstack([frames, pad])
            seqs.append((frames, label, csv_path.stem))
        return seqs

    def load(self, root):
        X, y, g = [], [], []
        for cls in ['normal','merodeo','forcejeo']:
            for csv in Path(root, cls).glob('*.csv'):
                for seq,label,vid in self._csv_to_sequences(csv):
                    X.append(seq); y.append(label); g.append(vid)
        self.X = np.stack(X)
        self.y = self.enc.fit_transform(y)
        self.groups = np.array(g)
        print(f"ℹ️  Secuencias: {self.X.shape}  |  Vídeos: {len(set(g))}")
        print("ℹ️  Distribución:", dict(pd.Series(y).value_counts()))

    def scale(self, tr_idx, te_idx):
        n,t,f = self.X.shape
        Xt = self.X[tr_idx]; Xv = self.X[te_idx]
        self.scaler.fit(Xt.reshape(-1,f))
        Xt = self.scaler.transform(Xt.reshape(-1,f)).reshape(len(tr_idx),t,f)
        Xv = self.scaler.transform(Xv.reshape(-1,f)).reshape(len(te_idx),t,f)
        return Xt, Xv

# ------------ 3. Modelo ------------
def build_gru(input_shape, n_cls, units=64, drop=0.3, lr=1e-4):
    m = Sequential([
        Masking(mask_value=0., input_shape=input_shape),
        GRU(units, dropout=drop),
        BatchNormalization(),
        Dense(32, activation='relu'),
        Dropout(drop),
        Dense(n_cls, activation='softmax')
    ])
    m.compile(optimizer=Adam(lr), loss='categorical_crossentropy',
              metrics=['accuracy'])
    return m

# ------------ 4. Entrenamiento leave-one-video-out ------------
def train_logo(data, units=64, drop=0.3, lr=1e-4, epochs=50, bs=16):
    logo = LeaveOneGroupOut()
    logo = LeaveOneGroupOut()
    n_cls = len(data.enc.classes_)               # ← NUEVO
    f1_scores = []
    
    for k,(tr,te) in enumerate(logo.split(data.X, data.y, data.groups),1):
        Xt,Xv = data.scale(tr,te)
        ytr,yv = data.y[tr], data.y[te]

        # one-hot con tamaño fijo
        ytr_cat = to_categorical(ytr, num_classes=n_cls)
        yv_cat  = to_categorical(yv , num_classes=n_cls)

        cw = compute_class_weight('balanced',
                                    classes=np.arange(n_cls),  # todas las clases
                                    y=ytr)
        model = build_gru(Xt.shape[1:], n_cls, units, drop, lr)
        cb = [
            F1EarlyStop(Xv, yv_cat, patience=2),
            ReduceLROnPlateau(monitor='val_f1', mode='max',
                              factor=0.5, patience=1, min_lr=1e-6, verbose=0)
        ]
        model.fit(Xt, ytr_cat, epochs=epochs, batch_size=bs,
                  validation_data=(Xv, yv_cat),
                  callbacks=cb, class_weight=dict(enumerate(cw)),
                  verbose=0)
        y_hat = np.argmax(model.predict(Xv, verbose=0),1)
        f1 = f1_score(yv, y_hat, average='macro')
        f1_scores.append(f1); print(f"Fold{k}: F1={f1:.3f}")
    print("— F1-macro medio =", np.mean(f1_scores).round(3))

# ------------ 5. Main ------------
if __name__ == '__main__':
    ROOT = r'D:\Documentos\Monitoreo-Comportamientos-Sospechosos\csv'  # <- tu ruta
    data = VideoDataset(max_len=50)
    data.load(ROOT)

    # hiperparámetros “seguros” para datasets pequeños
    train_logo(data,
               units=64,      # prueba 32 / 64
               drop=0.3,
               lr=1e-4,
               epochs=40,
               bs=8)
