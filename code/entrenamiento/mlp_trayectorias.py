# mlp_trayectorias.py
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization


# ---------- cargador de datos secuenciales ----------
class VideoDataset:
    def __init__(self, max_len=50):
        self.max_len = max_len
        self.scaler, self.enc = StandardScaler(), LabelEncoder()
        self.cols = ['Velocidad','Aceleracion','Linealidad',
                     'Circularidad','Zigzag','Densidad','Area_Trayectoria',
                     'Centroide_X','Centroide_Y']

    def _csv2seq(self, csv):
        df = pd.read_csv(csv); seqs=[]
        label = ('normal' if 'normal'in csv.stem else
                 'merodeo' if 'merodeo'in csv.stem else 'forcejeo')
        df = df.sort_values(['Objeto','Frame'])
        for _,traj in df.groupby('Objeto'):
            x = traj[self.cols].values.astype('float32')[:self.max_len]
            if len(x)<self.max_len:
                x = np.vstack([x, np.zeros((self.max_len-len(x), len(self.cols)), 'float32')])
            seqs.append((x, label, csv.stem))
        return seqs

    def load(self, root):
        X, y, g = [], [], []
        for cls in ['normal','merodeo','forcejeo']:
            for csv in Path(root, cls).glob('*.csv'):
                for seq,lbl,vid in self._csv2seq(csv):
                    X.append(seq); y.append(lbl); g.append(vid)
        self.X = np.stack(X)
        self.y = self.enc.fit_transform(y)
        self.g = np.array(g)
        print("Secuencias:", self.X.shape, "| vídeos únicos:", len(set(g)))

# ---------- agregar estadísticas ----------
def make_aggregates(X):
    return np.concatenate([X.mean(1), X.std(1),
                           X.max(1),  X.min(1)], axis=1)

# ---------- MLP ----------
# nuevo MLP
def build_mlp(input_dim, h1=128, h2=64, drop=0.3, lr=5e-4, n_cls=3):
    model = Sequential([
        Input(shape=(input_dim,)),
        BatchNormalization(),
        Dense(h1, activation='relu'),
        Dropout(drop),
        Dense(h2, activation='relu'),
        Dropout(drop),
        Dense(n_cls, activation='softmax')
    ])
    model.compile(optimizer=Adam(lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model            # ← ahora sí devuelve el modelo


# ---------- cross-validation ----------
def cv_mlp(ds, cfg, epochs=60, bs=16):
    X = make_aggregates(ds.X)
    n_cls = len(ds.enc.classes_)
    cv = StratifiedGroupKFold(6, shuffle=True, random_state=42)
    scores = []
    for k, (tr, te) in enumerate(cv.split(X, ds.y, ds.g), 1):

        # 1️⃣  Escalamos con stats del fold
        scaler = StandardScaler().fit(X[tr])
        Xtr = scaler.transform(X[tr])
        Xv  = scaler.transform(X[te])

        ytr, yv = ds.y[tr], ds.y[te]
        ytr_cat = to_categorical(ytr, n_cls)
        yv_cat  = to_categorical(yv,  n_cls)

        cw = compute_class_weight('balanced',
                              classes=np.arange(n_cls), y=ytr)
        mlp = build_mlp(X.shape[1], **cfg, n_cls=n_cls)

        mlp.fit(Xtr, ytr_cat,
                epochs=80, batch_size=32,
                validation_data=(Xv, yv_cat),   # sin aleatoriedad
                callbacks=[EarlyStopping(patience=8,
                                        restore_best_weights=True)],
                class_weight=dict(enumerate(cw)),
                verbose=0)

        y_hat = np.argmax(mlp.predict(Xv, verbose=0), 1)
        f1 = f1_score(yv, y_hat, average='macro')
        scores.append(f1)
        print(f"Fold{k}: F1={f1:.3f}")

    print("MLP F1-macro =", np.mean(scores).round(3),
        "±", np.std(scores).round(3))



# ---------- MAIN ----------
if __name__ == "__main__":
    ROOT = r"D:\Documentos\Monitoreo-Comportamientos-Sospechosos\csv"
    ds = VideoDataset(max_len=50); ds.load(ROOT)
    cfg = dict(h1=128, h2=64, drop=0.3, lr=1e-3)
    cv_mlp(ds, cfg)
