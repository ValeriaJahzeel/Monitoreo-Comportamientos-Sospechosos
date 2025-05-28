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
    def _csv_to_sequence(self, csv_path):
        df = pd.read_csv(csv_path)
        if df.empty: return None
        wanted = ['Velocidad','Aceleracion',
                  'Linealidad','Circularidad','Zigzag',
                  'Densidad','Area_Trayectoria',
                  'Centroide_X','Centroide_Y']
        cols   = [c for c in wanted if c in df.columns]
        seq    = []
        for f in sorted(df.Frame.unique()):
            fr = df[df.Frame==f]
            vec = [fr[c].mean() for c in cols]
            vec.append(fr['Objeto'].nunique())                      # num objetos
            vec.append((fr['En_Interaccion']==1).mean() if 'En_Interaccion' in fr else 0.)
            vec.append(float((fr['Es_Ciclico']==1).any()) if 'Es_Ciclico' in fr else 0.)
            seq.append(vec)
        return np.nan_to_num(np.array(seq,dtype=np.float32))

    # -- 2.2  cargar todas las secuencias -----------
    def load_data(self, root, max_len=50):
        X, y, g = [], [], []
        for cls in ['normal','merodeo','forcejeo']:
            for csv in Path(root,cls).glob('*.csv'):
                s = self._csv_to_sequence(csv)
                if s is None: continue
                s = s[:max_len] if len(s)>max_len else \
                    np.vstack([s, np.zeros((max_len-len(s), s.shape[1]))])
                X.append(s); y.append(cls); g.append(csv.stem)
        self.X = np.stack(X); self.y = np.array(y); self.groups=np.array(g)
        print(f"ℹ️  Datos: {self.X.shape}  (videos={len(self.X)})")
        print(f"ℹ️  Distribución: {dict(pd.Series(self.y).value_counts())}")

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
        model.add(Bidirectional(
            LSTM(cfg['lstm_units_1'],
                 return_sequences=False,
                 dropout=cfg['dropout'],
                 recurrent_dropout=0.3),
            input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(cfg['dropout']))
        model.add(Dense(n_classes, activation='softmax'))
        model.compile(optimizer=Adam(cfg['learning_rate']),
                      loss=focal_loss(alpha=0.25, gamma=2.0),
                      metrics=['accuracy'])
        return model

    # ------------ 3.  Grid Search K-fold -----------
    def grid_search(self, param_grid, epochs=60, val_split=0.15):
        combos = list(itertools.product(*param_grid.values()))
        best_f1, best_cfg = -1, None
        gkf = GroupKFold(n_splits=5)
        y_enc = self.encoder.fit_transform(self.y)
        n_classes = len(self.encoder.classes_)

        for cfg_vals in combos:
            cfg = dict(zip(param_grid.keys(), cfg_vals))
            fold_f1 = []
            print(f"\n🔧 Probando cfg: {cfg}")
            for k,(tr,te) in enumerate(gkf.split(self.X, y_enc, groups=self.groups)):
                Xtr,Xte = self._scale(tr,te)
                ytr, yte = y_enc[tr], y_enc[te]
                ytr_cat  = to_categorical(ytr,n_classes); yte_cat=to_categorical(yte,n_classes)

                model = self.build_model(cfg, Xtr.shape[1:], n_classes)

                cw = compute_class_weight('balanced',
                                          classes=np.unique(ytr),
                                          y=ytr)
                model.fit(Xtr, ytr_cat,
                          epochs=epochs,
                          batch_size=cfg['batch_size'],
                          validation_split=val_split,
                          callbacks=[EarlyStopping(patience=5, restore_best_weights=True,
                                                   monitor='val_loss')],
                          class_weight=dict(enumerate(cw)),
                          verbose=0)

                y_pred = np.argmax(model.predict(Xte,verbose=0),1)
                f1 = f1_score(yte, y_pred, average='macro')
                fold_f1.append(f1); print(f"   Fold{k+1}: F1={f1:.3f}")

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
    'lstm_units_1': [32, 64],
    'dropout': [0.4],
    'learning_rate': [0.0005],
    'batch_size': [4]
    }
    best_cfg = vid.grid_search(param_grid, epochs=60)
    vid.train_final(best_cfg, epochs=120, val_split=0.25)

    # ----  Informe final
    print("\n🛈 Entrenado con configuración óptima:", best_cfg)
