"""
LSTM v3 – clasificación de secuencias de vídeo
Autor: tú | mayo-2025
────────────────────────────────────────────
Requisitos: tensorflow-keras ≥2.11, scikit-learn ≥1.3, matplotlib, seaborn, pandas
Ajusta la variable ROOT con la ruta que contiene las carpetas: normal / merodeo / forcejeo
"""

import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns, joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.layers import Input, Masking, Bidirectional, LSTM, BatchNormalization, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import tensorflow as tf, warnings
warnings.filterwarnings("ignore")
tf.random.set_seed(42); np.random.seed(42)

ROOT = r"D:\Documentos\Monitoreo-Comportamientos-Sospechosos\csv"
MAX_LEN = 50
BATCH = 8
EPOCHS = 120
PATIENCE_F1 = 8
LR = 5e-4

# ──────────── 1. Data loader ────────────
COLS = ['Velocidad','Aceleracion','Cambio_Direccion','Linealidad',
        'Circularidad','Zigzag','Densidad','Area_Trayectoria',
        'Centroide_X','Centroide_Y']

def csv2seq(csv):
    df = pd.read_csv(csv)
    lbl = ('normal' if 'normal'in csv.stem else
           'merodeo'if 'merodeo'in csv.stem else 'forcejeo')
    seq = []
    for frame, fr_data in df.groupby('Frame'):
        feats = [ fr_data[c].dropna().mean() if c in fr_data else 0.0 for c in COLS ]
        # n_objetos
        feats.append(fr_data['Objeto'].nunique() if 'Objeto' in fr_data else len(fr_data))
        # pct_interacción
        feats.append((fr_data['En_Interaccion']==1).mean() if 'En_Interaccion' in fr_data else 0.0)
        # es_cíclico (flag)
        feats.append(float((fr_data['Es_Ciclico']==1).any()) if 'Es_Ciclico' in fr_data else 0.0)
        seq.append(feats)
    if not seq: return None,None,None
    seq = np.nan_to_num(np.array(seq,dtype='float32'))
    return seq, lbl, csv.stem

X, y, g = [],[],[]
for cls in ['normal','merodeo','forcejeo']:
    for csv in Path(ROOT,cls).glob("*.csv"):
        s,l,v = csv2seq(csv)
        if s is not None:
            X.append(s); y.append(l); g.append(v)
X = np.array(X, dtype=object); y = np.array(y); g = np.array(g)
enc = LabelEncoder(); y_enc = enc.fit_transform(y)
print("Vídeos:", len(X), "| clases:", dict(pd.Series(y).value_counts()))

# ─ padding / truncating a MAX_LEN ─
F = X[0].shape[1]
def pad(seq):
    if len(seq)>MAX_LEN: return seq[:MAX_LEN]
    if len(seq)<MAX_LEN:
        pad = np.zeros((MAX_LEN-len(seq),F),dtype='float32')
        return np.vstack([seq,pad])
    return seq
X_pad = np.stack([pad(s) for s in X])

# ──────────── 2. Métrica macro-F1 callback ────────────
class MacroF1(Callback):
    def __init__(self,X_val,y_val,patience=PATIENCE_F1):
        super().__init__(); self.Xv=X_val; self.yv=y_val
        self.best=0; self.wait=0; self.pat=patience
    def on_epoch_end(self,epoch,logs=None):
        y_hat=np.argmax(self.model.predict(self.Xv,verbose=0),1)
        f1=f1_score(self.yv,y_hat,average='macro')
        logs['val_macroF1']=f1; print(f" — val_F1={f1:.3f}")
        if f1>self.best+1e-3: self.best=f1; self.wait=0
        else:
            self.wait+=1
            if self.wait>=self.pat:
                print(f"⏹ Early stop (no mejora F1)"); self.model.stop_training=True

# ──────────── 3. Modelo ────────────
def build_lstm(input_shape,n_cls):
    m = Sequential([
        Input(shape=input_shape),
        Masking(mask_value=0.0),
        Bidirectional(LSTM(64, dropout=0.4, recurrent_dropout=0.4)),
        BatchNormalization(),
        Dense(64,'relu'), Dropout(0.4),
        Dense(n_cls,'softmax')
    ])
    m.compile(Adam(LR),'categorical_crossentropy',metrics=['accuracy'])
    return m

# ──────────── 4. 6-fold CV para estimar F1 ────────────
cv = StratifiedGroupKFold(n_splits=6,shuffle=True,random_state=42)
f1_scores=[]

for k,(tr,te) in enumerate(cv.split(X_pad,y_enc,g),1):
    scaler = StandardScaler().fit(X_pad[tr].reshape(-1,F))
    Xt = scaler.transform(X_pad[tr].reshape(-1,F)).reshape(len(tr),MAX_LEN,F)
    Xv = scaler.transform(X_pad[te].reshape(-1,F)).reshape(len(te),MAX_LEN,F)
    ytr,yv = y_enc[tr], y_enc[te]
    ytr_cat,yv_cat = to_categorical(ytr), to_categorical(yv)
    cw = compute_class_weight('balanced',classes=np.arange(len(enc.classes_)),y=ytr)
    model = build_lstm((MAX_LEN,F), len(enc.classes_))
    cb=[MacroF1(Xv,yv), ReduceLROnPlateau(monitor='val_macroF1',mode='max',
                                          factor=0.5,patience=4,min_lr=1e-6,verbose=0)]
    model.fit(Xt,ytr_cat,epochs=EPOCHS,batch_size=BATCH,
              validation_data=(Xv,yv_cat),class_weight=dict(enumerate(cw)),
              verbose=0,callbacks=cb)
    y_hat = np.argmax(model.predict(Xv,verbose=0),1)
    f1 = f1_score(yv,y_hat,average='macro'); f1_scores.append(f1)
    print(f"Fold{k}: F1={f1:.3f}")

print("\n⚑ F1 CV =", np.mean(f1_scores).round(3), "±", np.std(f1_scores).round(3))

# ──────────── 5. Split final 80/20 hold-out ────────────
X_train,X_test,y_train,y_test,groups_train,_ = train_test_split(
    X_pad,y_enc,g,test_size=0.20,stratify=y_enc,random_state=42)

scaler = StandardScaler().fit(X_train.reshape(-1,F))
X_train = scaler.transform(X_train.reshape(-1,F)).reshape(len(X_train),MAX_LEN,F)
X_test  = scaler.transform(X_test.reshape(-1,F)).reshape(len(X_test),MAX_LEN,F)
ytr_cat, yte_cat = to_categorical(y_train), to_categorical(y_test)

cw = compute_class_weight('balanced',classes=np.arange(len(enc.classes_)),y=y_train)
final_lstm = build_lstm((MAX_LEN,F), len(enc.classes_))
hist = final_lstm.fit(
    X_train,ytr_cat,epochs=EPOCHS,batch_size=BATCH,
    validation_split=0.15,
    callbacks=[MacroF1(X_test,y_test),ReduceLROnPlateau(monitor='val_macroF1',
              mode='max',factor=0.5,patience=5,min_lr=1e-6,verbose=1)],
    class_weight=dict(enumerate(cw)),verbose=1)

# ──────────── 6. Evaluación hold-out + gráficas ────────────
y_pred = np.argmax(final_lstm.predict(X_test),1)
f1_hold = f1_score(y_test,y_pred,average='macro')
print(f"\n▶ F1 hold-out: {f1_hold:.3f}")

plt.figure(figsize=(5,4))
plt.plot(hist.history['loss']); plt.plot(hist.history['val_loss'])
plt.title('Loss'); plt.xlabel('Epoch'); plt.legend(['train','val']); plt.tight_layout(); plt.show()

plt.figure(figsize=(5,4))
plt.plot(hist.history['accuracy']); plt.plot(hist.history['val_accuracy'])
plt.title('Accuracy'); plt.xlabel('Epoch'); plt.legend(['train','val']); plt.tight_layout(); plt.show()

cm = confusion_matrix(y_test,y_pred,normalize='true')
disp = ConfusionMatrixDisplay(cm,display_labels=enc.classes_)
disp.plot(cmap='Blues',colorbar=False); plt.tight_layout(); plt.show()

# ──────────── 7. Guardar modelo + scaler ────────────
final_lstm.save('lstm_v3_final.h5')
joblib.dump({'scaler':scaler,'label_encoder':enc,'max_len':MAX_LEN},
            'prepro_lstm_v3.pkl')
print("💾 Modelo y preprocesamiento guardados.")
