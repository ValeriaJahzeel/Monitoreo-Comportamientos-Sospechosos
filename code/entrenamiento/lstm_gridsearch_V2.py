"""
Detector de merodeo/forcejeo - versión mejorada
Autor: Tú
"""

# -------------  Imports y utilidades -------------
import numpy as np, pandas as pd, os, warnings, itertools
from pathlib import Path
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection import LeaveOneGroupOut

# from iterstrat.ml_stratifiers import MultilabelStratifiedGroupKFold as MSGKF


# ------------ 1.  Focal Loss -------------
def focal_loss(alpha=0.25, gamma=2.0):
    def loss(y_true, y_pred):
        y_true = K.cast(y_true, "float32")
        ce   = K.categorical_crossentropy(y_true, y_pred)
        probs= K.sum(y_true * y_pred, axis=-1)
        fl   = alpha * K.pow(1. - probs, gamma) * ce
        return fl
    return loss

# ------------ 2.  Clase principal -------------
class VideoLSTM:
    def __init__(self, random_state=42):
        self.rng  = np.random.RandomState(random_state)
        self.scaler = StandardScaler()
        self.encoder= LabelEncoder()

    # -- 2.1  leer CSV → secuencia -----------------
    def _csv_to_sequences_per_object(self, csv_path):
        """
        Devuelve una lista de (sequence, label, video_name) —una por OBJETO (persona).
        Cada secuencia: máx 50 frames, con padding al final si es más corta.
        """
        df = pd.read_csv(csv_path)
        if df.empty: 
            return []

        video_name = Path(csv_path).stem
        label      = 'normal' if 'normal'   in csv_path.stem else \
                    'merodeo' if 'merodeo' in csv_path.stem else 'forcejeo'

        wanted = ['Velocidad','Aceleracion','Cambio_Direccion',
                'Linealidad','Circularidad','Zigzag',
                'Densidad','Area_Trayectoria',
                'Centroide_X','Centroide_Y']
        cols = [c for c in wanted if c in df.columns]

        seqs = []
        max_len = 50
        for obj_id, traj in df.groupby('Objeto'):
            traj = traj.sort_values('Frame')
            frames = []
            for _, row in traj.iterrows():
                vec = [row[c] for c in cols]
                frames.append(vec)
                if len(frames) == max_len:
                    break
            # padding
            if len(frames) < max_len:
                pad = np.zeros((max_len - len(frames), len(cols)), dtype=np.float32)
                frames = np.vstack([frames, pad])
            else:
                frames = np.array(frames, dtype=np.float32)

            frames = np.nan_to_num(frames)
            seqs.append((frames, label, video_name))  # tuple
        return seqs

    # -- 2.2  cargar todas las secuencias -----------
    def load_data(self, root, max_len=50):
        X, y, g = [], [], []
        root = Path(root)
        for cls in ['normal','merodeo','forcejeo']:
            for csv in (root/cls).glob('*.csv'):
                obj_seqs = self._csv_to_sequences_per_object(csv)
                for seq, lbl, vid in obj_seqs:
                    X.append(seq); y.append(lbl); g.append(vid)

        self.X = np.stack(X)
        self.y = np.array(y)
        self.groups = np.array(g)

        print(f"ℹ️  Trayectorias: {self.X.shape}  (videos={len(set(g))})")
        print(f"ℹ️  Distrib. clases: {dict(pd.Series(self.y).value_counts())}")


    # -- 2.3  normalizar ----------------------------
    def _scale(self, train_idx, test_idx):
        Xt, Xv = self.X[train_idx], self.X[test_idx]
        n_t, t, f = Xt.shape
        self.scaler.fit(Xt.reshape(-1,f))
        Xt = self.scaler.transform(Xt.reshape(-1,f)).reshape(n_t,t,f)
        
        # si test está vacío, salimos
        if len(test_idx) == 0:
            return Xt, None   
        
        Xv = self.X[test_idx]
        n_v = len(test_idx)
        Xv = self.scaler.transform(Xv.reshape(-1, f)).reshape(n_v, t, f)
        return Xt, Xv
    

    # -- 2.4  construir modelo ----------------------
    def build_model(self, cfg, input_shape, n_classes):
        model = Sequential()
        model.add(Masking(mask_value=0., input_shape=input_shape))     # NUEVO
        model.add(Bidirectional(
            LSTM(cfg['lstm_units_1'],
                return_sequences=False,
                dropout=cfg['dropout'])))
        model.add(BatchNormalization())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(cfg['dropout']))
        model.add(Dense(n_classes, activation='softmax'))
        model.compile(optimizer=Adam(cfg['learning_rate']),
                    loss='categorical_crossentropy',                # CAMBIO
                    metrics=['accuracy'])
        return model


    # ------------ 3.  Grid Search K-fold -------------
    def grid_search(self, param_grid, epochs=60, val_split=0.15):
        combos = list(itertools.product(*param_grid.values()))
        best_f1, best_cfg = -1, None

        # ❶  Ajusta el encoder ANTES de imprimir las clases
        y_enc = self.encoder.fit_transform(self.y)
        print("Índices de clase:", dict(enumerate(self.encoder.classes_)))
        n_classes = len(self.encoder.classes_)

        splitter = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=42)

        for cfg_vals in combos:
            cfg = dict(zip(param_grid.keys(), cfg_vals))
            fold_f1 = []
            print(f"\n🔧 Probando cfg: {cfg}")

            # ❷  Vuelve a enumerar los folds para usar k en el print
            for k, (tr, te) in enumerate(
                    splitter.split(self.X, y_enc, groups=self.groups), 1):
                Xtr, Xte = self._scale(tr, te)
                ytr, yte = y_enc[tr], y_enc[te]
                ytr_cat, yte_cat = to_categorical(ytr, n_classes), to_categorical(yte, n_classes)

                model = self.build_model(cfg, Xtr.shape[1:], n_classes)
                cw = compute_class_weight('balanced', classes=np.unique(ytr), y=ytr)

                model.fit(
                    Xtr, ytr_cat,
                    epochs=epochs,
                    batch_size=cfg['batch_size'],
                    validation_split=val_split,
                    callbacks=[EarlyStopping(patience=5, restore_best_weights=True,
                                            monitor='val_loss')],
                    class_weight=dict(enumerate(cw)),
                    verbose=0
                )

                y_pred = np.argmax(model.predict(Xte, verbose=0), 1)
                f1 = f1_score(yte, y_pred, average='macro')
                fold_f1.append(f1)
                print(f"   Fold{k}: F1={f1:.3f}")

            avg_f1 = np.mean(fold_f1)
            print(f"➡️  F1-macro medio = {avg_f1:.3f}")
            if avg_f1 > best_f1:
                best_f1, best_cfg = avg_f1, cfg

        print(f"\n🏆 Mejor cfg: {best_cfg} (F1-macro={best_f1:.3f})")
        return best_cfg


    # ------------ 4.  Entrenar final --------------
    def train_final(self, cfg, epochs=120, val_split=0.25):
        y_enc = self.encoder.transform(self.y)
        n_cls = len(self.encoder.classes_)
        train_idx = np.arange(len(self.X))   # usamos todo para entrenar
        Xtr,_ = self._scale(train_idx, [])
        ytr_cat = to_categorical(y_enc, n_cls)

        self.model = self.build_model(cfg, Xtr.shape[1:], n_cls)
        cw = compute_class_weight('balanced', classes=np.unique(y_enc), y=y_enc)

        hist = self.model.fit(
            Xtr, ytr_cat,
            epochs=epochs,
            batch_size=cfg['batch_size'],
            validation_split=val_split,
            callbacks=[
                EarlyStopping(patience=4, 
                              min_delta=0.005,
                              restore_best_weights=True,
                              monitor='val_loss'),
                ReduceLROnPlateau(monitor='val_loss',
                              factor=0.5,
                              patience=3,
                              min_lr=1e-6)
                ],
            class_weight=dict(enumerate(cw)),
            verbose=1)
        self.history = hist.history
        print("✅ Entrenamiento final completado")

# ----------- 5.  Ejecución sencilla -------------
if __name__ == "__main__":
    # root = input("📂 Carpeta con CSVs por clase: ").strip()
    root = r'D:\Documentos\Monitoreo-Comportamientos-Sospechosos\csv'
    vid = VideoLSTM(random_state=42)
    vid.load_data(root, max_len=50)

    # -- Grid Search reducido
    param_grid = {
        'lstm_units_1': [32, 64, 128],
        'dropout'     : [0.3, 0.4],
        'learning_rate': [1e-4, 5e-4],
        'batch_size'   : [8]
    }
    best_cfg = vid.grid_search(param_grid, epochs=60)
    vid.train_final(best_cfg, epochs=120, val_split=0.25)

    # ----  Informe final
    print("\n🛈 Entrenado con configuración óptima:", best_cfg)
