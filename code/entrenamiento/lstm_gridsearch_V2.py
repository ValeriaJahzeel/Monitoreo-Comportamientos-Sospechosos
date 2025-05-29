# lstm_aug_trayectorias.py
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Masking, LSTM, BatchNormalization, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# ---------- dataset secuencial ----------
class VideoDataset:
    def __init__(self,max_len=50):
        self.max_len=max_len
        self.scaler, self.enc = StandardScaler(), LabelEncoder()
        self.cols=['Velocidad','Aceleracion','Linealidad',
                   'Circularidad','Zigzag','Densidad','Area_Trayectoria',
                   'Centroide_X','Centroide_Y']

    def _csv2seq(self,csv):
        df=pd.read_csv(csv); seqs=[]
        label=('normal'if 'normal'in csv.stem else
               'merodeo'if 'merodeo'in csv.stem else 'forcejeo')
        for _,traj in df.groupby('Objeto'):
            x=traj.sort_values('Frame')[self.cols].values.astype('float32')[:self.max_len]
            if len(x)<self.max_len:
                x=np.vstack([x,np.zeros((self.max_len-len(x),len(self.cols)),'float32')])
            seqs.append((x,label,csv.stem))
        return seqs

    def load(self,root):
        X,y,g=[],[],[]
        for cls in ['normal','merodeo','forcejeo']:
            for csv in Path(root,cls).glob('*.csv'):
                for s,l,v in self._csv2seq(csv):
                    X.append(s); y.append(l); g.append(v)
        self.X=np.stack(X); self.y=self.enc.fit_transform(y); self.g=np.array(g)
        print("Secuencias:",self.X.shape,"| vídeos únicos:",len(set(g)))

    def scale(self,tr,te):
        n,t,f=self.X.shape
        Xt,Xv=self.X[tr],self.X[te]
        self.scaler.fit(Xt.reshape(-1,f))
        Xt=self.scaler.transform(Xt.reshape(-1,f)).reshape(len(tr),t,f)
        Xv=self.scaler.transform(Xv.reshape(-1,f)).reshape(len(te),t,f)
        return Xt,Xv

# ---------- generador con augment ----------
def augment_batch(x, noise_std=0.01):
    x = x.copy()
    x += np.random.normal(0, noise_std, x.shape).astype('float32')
    mask = np.random.rand(len(x)) < 0.3
    x[mask] = x[mask, ::-1, :]
    return x

class AugmentSeq(Sequence):
    def __init__(self,X,y,bs=8): self.X,self.y,self.bs=X,y,bs
    def __len__(self): return int(np.ceil(len(self.X)/self.bs))
    def __getitem__(self,i):
        sl=slice(i*self.bs,(i+1)*self.bs)
        return augment_batch(self.X[sl]), self.y[sl]

# ---------- LSTM compacto ----------
def build_lstm(inshape,n_cls,units=32,drop=0.4,lr=1e-4):
    m=Sequential([
        Masking(mask_value=0.,input_shape=inshape),
        LSTM(units,dropout=drop,recurrent_dropout=0.2),
        BatchNormalization(),
        Dense(32,'relu'),
        Dropout(drop),
        Dense(n_cls,'softmax')
    ])
    m.compile(Adam(lr),'categorical_crossentropy',metrics=['accuracy'])
    return m

# ---------- cross-validation ----------
def cv_lstm(ds,epochs=50,bs=8):
    n_cls=len(ds.enc.classes_)
    cv=StratifiedGroupKFold(6,shuffle=True,random_state=42)
    scores=[]
    for k,(tr,te) in enumerate(cv.split(ds.X, ds.y, ds.g),1):
        Xt,Xv = ds.scale(tr,te)
        ytr,yv = ds.y[tr], ds.y[te]
        ytr_cat,to_cat = to_categorical(ytr,n_cls), to_categorical(yv,n_cls)
        cw = compute_class_weight('balanced',classes=np.arange(n_cls),y=ytr)
        model=build_lstm(Xt.shape[1:],n_cls)
        gen=AugmentSeq(Xt,ytr_cat,bs)
        model.fit(gen,epochs=epochs,
                  validation_data=(Xt[::5],ytr_cat[::5]),
                  callbacks=[EarlyStopping(patience=5,restore_best_weights=True)],
                  class_weight=dict(enumerate(cw)),verbose=0)
        y_hat=np.argmax(model.predict(Xv,verbose=0),1)
        f1=f1_score(yv,y_hat,average='macro'); scores.append(f1)
        print(f"Fold{k}: F1={f1:.3f}")
    print("LSTM F1-macro =",np.mean(scores).round(3),"±",np.std(scores).round(3))

# ---------- MAIN ----------
if __name__ == "__main__":
    ROOT = r"D:\Documentos\Monitoreo-Comportamientos-Sospechosos\csv"
    ds = VideoDataset(max_len=50); ds.load(ROOT)
    cv_lstm(ds, epochs=50, bs=8)
