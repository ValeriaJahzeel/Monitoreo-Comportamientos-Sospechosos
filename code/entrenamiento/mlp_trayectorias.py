#!/usr/bin/env python
# mlp_trayectorias_final.py
import numpy as np, pandas as pd, joblib, matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, BatchNormalization, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

# ---------------- configuración ----------------
ROOT = r"D:\Documentos\Monitoreo-Comportamientos-Sospechosos\csv"
MAX_LEN = 50
COLS = ['Velocidad','Aceleracion','Linealidad',
        'Circularidad','Zigzag','Densidad','Area_Trayectoria',
        'Centroide_X','Centroide_Y']
CFG = dict(h1=128, h2=64, drop=0.30, lr=5e-4, batch=32, epochs=120, patience=12)

# ---------------- carga de datos ----------------
def csv2seq(csv):
    df = pd.read_csv(csv)
    lbl = ('normal' if 'normal' in csv.stem else
           'merodeo' if 'merodeo' in csv.stem else 'forcejeo')
    for _, traj in df.groupby('Objeto'):
        x = traj.sort_values('Frame')[COLS].values.astype('float32')[:MAX_LEN]
        if len(x) < MAX_LEN:
            x = np.vstack([x, np.zeros((MAX_LEN-len(x), len(COLS)), 'float32')])
        yield x, lbl, csv.stem

X, y, g = [], [], []
for cls in ['normal','merodeo','forcejeo']:
    for csv in Path(ROOT, cls).glob('*.csv'):
        for seq, lbl, vid in csv2seq(csv):
            X.append(seq); y.append(lbl); g.append(vid)
X = np.stack(X); y = np.array(y); g = np.array(g)
enc = LabelEncoder(); y_enc = enc.fit_transform(y)
print(f"Secuencias: {X.shape} | videos: {len(set(g))}")

# ---------------- agregados ----------------
X_agg = np.concatenate([X.mean(1), X.std(1), X.max(1), X.min(1)], axis=1)

# ---------------- train / hold-out split ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X_agg, y_enc, test_size=0.20, stratify=y_enc, random_state=42)

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train); X_test = scaler.transform(X_test)
n_cls = len(enc.classes_)

# ---------------- modelo MLP ----------------
def build_mlp(input_dim, n_cls, cfg):
    model = Sequential([
        Input(shape=(input_dim,)),
        BatchNormalization(),
        Dense(cfg['h1'], activation='relu'),
        Dropout(cfg['drop']),
        Dense(cfg['h2'], activation='relu'),
        Dropout(cfg['drop']),
        Dense(n_cls, activation='softmax')
    ])
    model.compile(Adam(cfg['lr']), 'categorical_crossentropy',
                  metrics=['accuracy'])
    return model

ytr_cat = to_categorical(y_train, n_cls); yte_cat = to_categorical(y_test, n_cls)
cw = compute_class_weight(class_weight='balanced',
                          classes=np.arange(n_cls), y=y_train)

mlp = build_mlp(X_train.shape[1], n_cls, CFG)
hist = mlp.fit(
    X_train, ytr_cat,
    validation_split=0.15,
    epochs=CFG['epochs'],
    batch_size=CFG['batch'],
    class_weight=dict(enumerate(cw)),
    callbacks=[EarlyStopping(patience=CFG['patience'],
                             restore_best_weights=True)],
    verbose=1)

# ---------------- evaluación ----------------
y_pred = np.argmax(mlp.predict(X_test), 1)
f1 = f1_score(y_test, y_pred, average='macro')
print(f"► F1-macro hold-out: {f1:.3f}")

# ---------------- gráficas ----------------
plt.figure(figsize=(5,4))
plt.plot(hist.history['loss'], label='train'); plt.plot(hist.history['val_loss'], label='val')
plt.title('Curva de pérdida'); plt.xlabel('Época'); plt.ylabel('Loss'); plt.legend(); plt.tight_layout()
plt.show()

plt.figure(figsize=(5,4))
plt.plot(hist.history['accuracy'], label='train'); plt.plot(hist.history['val_accuracy'], label='val')
plt.title('Exactitud'); plt.xlabel('Época'); plt.ylabel('Accuracy'); plt.legend(); plt.tight_layout()
plt.show()

cm = confusion_matrix(y_test, y_pred, normalize='true')
disp = ConfusionMatrixDisplay(cm, display_labels=enc.classes_)
disp.plot(cmap='Blues', colorbar=False); plt.title('Matriz de confusión (normalizada)'); plt.tight_layout(); plt.show()

# ---------------- guardado ----------------
mlp.save('mlp_trayectorias_final.h5')
joblib.dump({'scaler': scaler, 'label_encoder': enc},
            'preprocesamiento_agregados.pkl')
print("Modelo y escalador guardados.")
