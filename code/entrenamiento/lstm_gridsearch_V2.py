"""
Detector merodeo/forcejeo  ·  CV estratificado por vídeo
Autor: tú
────────────────────────────────────────────────────────
Requiere: scikit-learn ≥1.3   ·   TensorFlow ≥2.11
Si tu scikit-learn es más viejo, usa `pip install iterative-stratification`
y cambia StratifiedGroupKFold por MultilabelStratifiedGroupKFold.
"""

# ─────────────────── imports
import numpy as np, pandas as pd, warnings, itertools
from pathlib import Path; warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier

# ─────────────────── callback F1
class F1EarlyStop(Callback):
    def __init__(self, Xv, yv_cat, patience=2):
        super().__init__(); self.Xv=Xv; self.yv=yv_cat
        self.best=0; self.wait=0; self.pat=patience
    def on_epoch_end(self, epoch, logs=None):
        y_pred=np.argmax(self.model.predict(self.Xv,verbose=0),1)
        y_true=np.argmax(self.yv,1)
        f1=f1_score(y_true,y_pred,average='macro')
        logs['val_f1']=f1; print(f"  — val_F1={f1:.3f}")
        if f1>self.best+1e-3: self.best,f1,self.wait=f1, f1, 0
        else:
            self.wait+=1
            if self.wait>=self.pat:
                print(f"\n⏹ Early stop en epoch {epoch+1}"); self.model.stop_training=True

# ─────────────────── dataset helper
class VideoDataset:
    def __init__(self,max_len=50,cols=None):
        self.max_len=max_len
        self.wanted=cols or [
            'Velocidad','Aceleracion','Cambio_Direccion','Linealidad',
            'Circularidad','Zigzag','Densidad','Area_Trayectoria',
            'Centroide_X','Centroide_Y']
        self.scaler,self.enc=StandardScaler(),LabelEncoder()

    def _csv2seq(self,csv):
        df=pd.read_csv(csv);   seqs=[]
        if df.empty: return seqs
        lbl=('normal' if 'normal'in csv.stem else
             'merodeo'if 'merodeo'in csv.stem else 'forcejeo')
        cols=[c for c in self.wanted if c in df.columns]
        for _,traj in df.groupby('Objeto'):
            fr=traj.sort_values('Frame')[cols].values.astype('float32')[:self.max_len]
            if len(fr)<self.max_len:
                fr=np.vstack([fr,np.zeros((self.max_len-len(fr),len(cols)),'float32')])
            seqs.append((fr,lbl,csv.stem))
        return seqs

    def load(self,root):
        X,y,g=[],[],[]
        for cls in ['normal','merodeo','forcejeo']:
            for csv in Path(root,cls).glob('*.csv'):
                for s,l,v in self._csv2seq(csv):
                    X.append(s); y.append(l); g.append(v)
        self.X=np.stack(X); self.y=self.enc.fit_transform(y); self.groups=np.array(g)
        print(f"ℹ️ Secuencias {self.X.shape} | Vídeos {len(set(g))}")
        print("ℹ️ Distribución:",dict(pd.Series(y).value_counts()))

    def scale(self,tr,te):
        n,t,f=self.X.shape
        Xt,Xv=self.X[tr],self.X[te]
        self.scaler.fit(Xt.reshape(-1,f))
        Xt=self.scaler.transform(Xt.reshape(-1,f)).reshape(len(tr),t,f)
        Xv=self.scaler.transform(Xv.reshape(-1,f)).reshape(len(te),t,f)
        return Xt,Xv

# ─────────────────── modelo GRU
def build_gru(inshape,n_cls,units=64,drop=0.3,lr=1e-4):
    m=Sequential([
        Masking(mask_value=0.,input_shape=inshape),
        GRU(units,dropout=drop),
        BatchNormalization(),
        Dense(32,activation='relu'),
        Dropout(drop),
        Dense(n_cls,activation='softmax')])
    m.compile(optimizer=Adam(lr),loss='categorical_crossentropy',
              metrics=['accuracy'])
    return m

# ─────────────────── cross-validation GRU
def cv_gru(data,units=64,drop=0.3,lr=1e-4,epochs=40,bs=8):
    splitter=StratifiedGroupKFold(n_splits=6,shuffle=True,random_state=42)
    n_cls=len(data.enc.classes_)
    scores=[]
    for k,(tr,te) in enumerate(splitter.split(data.X,data.y,data.groups),1):
        Xt,Xv=data.scale(tr,te)
        ytr,yv=data.y[tr],data.y[te]
        ytr_cat=to_categorical(ytr,num_classes=n_cls)
        yv_cat =to_categorical(yv ,num_classes=n_cls)
        cw=compute_class_weight('balanced',classes=np.arange(n_cls),y=ytr)
        model=build_gru(Xt.shape[1:],n_cls,units,drop,lr)
        cb=[F1EarlyStop(Xv,yv_cat,patience=2),
            ReduceLROnPlateau(monitor='val_f1',mode='max',
                              factor=0.5,patience=1,min_lr=1e-6,verbose=0)]
        model.fit(Xt,ytr_cat,epochs=epochs,batch_size=bs,
                  validation_data=(Xv,yv_cat),callbacks=cb,
                  class_weight=dict(enumerate(cw)),verbose=0)
        y_hat=np.argmax(model.predict(Xv,verbose=0),1)
        f1=f1_score(yv,y_hat,average='macro'); scores.append(f1)
        print(f"Fold{k}: F1={f1:.3f}")
    print("— GRU F1-macro =",np.mean(scores).round(3),"±",np.std(scores).round(3))

# ─────────────────── baseline Random Forest
def rf_baseline(data):
    Xagg=np.concatenate([data.X.mean(1), data.X.std(1),
                         data.X.max(1) , data.X.min(1)],axis=1)
    splitter=StratifiedGroupKFold(n_splits=6,shuffle=True,random_state=42)
    scores=[]
    for tr,te in splitter.split(Xagg,data.y,data.groups):
        rf=RandomForestClassifier(
            n_estimators=400,class_weight='balanced',random_state=42)
        rf.fit(Xagg[tr],data.y[tr])
        y_hat=rf.predict(Xagg[te])
        scores.append(f1_score(data.y[te],y_hat,average='macro'))
    print("— RF F1-macro    =",np.mean(scores).round(3),"±",np.std(scores).round(3))

# ─────────────────── MAIN
if __name__=="__main__":
    ROOT=r'D:\Documentos\Monitoreo-Comportamientos-Sospechosos\csv'  # ← ajusta tu ruta
    ds=VideoDataset(max_len=50); ds.load(ROOT)

    print("\n🟢 Random Forest baseline")
    rf_baseline(ds)

    print("\n🔵 GRU (64u, drop0.3, lr1e-4)")
    cv_gru(ds,units=64,drop=0.3,lr=1e-4,epochs=40,bs=8)
