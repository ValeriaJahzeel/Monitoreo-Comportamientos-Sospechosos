#!/usr/bin/env python
"""
Ablation de features con Random Forest
• compara 'básico' vs 'extendido'
• muestra importancias por permutación
Autor: tú · mayo-2025
"""

import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns, joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.inspection import permutation_importance
from warnings import filterwarnings
filterwarnings("ignore", category=UserWarning)

# ───────────────────────────────────────
ROOT = r"D:\Documentos\Monitoreo-Comportamientos-Sospechosos\csv"
MAX_LEN = 50
COLS_BASIC = ['Velocidad','Aceleracion','Cambio_Direccion','Linealidad',
              'Circularidad','Zigzag','Densidad','Area_Trayectoria',
              'Centroide_X','Centroide_Y']
COLS_EXTRA = ['Desplazamiento','Direccion','Postura',
              'Patron_Movimiento','Frecuencia_Ciclo','Amplitud_Ciclo']
META_COLS  = ['n_objetos','pct_inter','es_ciclico']
# final listas
BASIC   = COLS_BASIC + META_COLS
EXTENDED= COLS_BASIC + COLS_EXTRA + META_COLS
# ───────────────────────────────────────

def csv2seq(csv):
    df = pd.read_csv(csv)
    def get(col, fr): 
        return fr[col].dropna().mean() if col in fr else 0.0
    seq=[]
    for frame, fr in df.groupby('Frame'):
        feats=[ get(c,fr) for c in COLS_BASIC+COLS_EXTRA ]
        # meta-features
        feats += [
            fr['Objeto'].nunique() if 'Objeto' in fr else len(fr),                 # n_objetos
            (fr['En_Interaccion']==1).mean() if 'En_Interaccion' in fr else 0.0,   # pct_inter
            float((fr['Es_Ciclico']==1).any()) if 'Es_Ciclico' in fr else 0.0      # es_ciclico
        ]
        seq.append(feats)
    if not seq: return None,None,None
    seq=np.nan_to_num(np.array(seq,dtype='float32'))
    lbl=('normal' if 'normal'in csv.stem else
         'merodeo'if 'merodeo'in csv.stem else 'forcejeo')
    return seq,lbl,csv.stem

X,y,g=[],[],[]
for cls in ['normal','merodeo','forcejeo']:
    for csv in Path(ROOT,cls).glob("*.csv"):
        s,l,v = csv2seq(csv)
        if s is not None: X.append(s); y.append(l); g.append(v)
X=np.array(X,dtype=object); y=np.array(y); g=np.array(g)
enc=LabelEncoder(); y_enc=enc.fit_transform(y)
print("Vídeos:",len(X)," | Clases:",dict(pd.Series(y).value_counts()))

def pad(seq,f):
    if len(seq)>MAX_LEN: return seq[:MAX_LEN]
    if len(seq)<MAX_LEN:
        return np.vstack([seq,np.zeros((MAX_LEN-len(seq),f),'float32')])
    return seq

F=len(BASIC)+len(COLS_EXTRA) # máx features por frame
X_pad=np.stack([pad(s,F) for s in X])

# ───────── aggregate helpers ─────────
def make_aggs(X_pad, cols):
    idx = [ (COLS_BASIC+COLS_EXTRA+META_COLS).index(c) for c in cols ]
    Xf = X_pad[:,:,idx]                       # (N,T,F_sel)
    mean = Xf.mean(1); std=Xf.std(1)
    mx = Xf.max(1); mn=Xf.min(1)
    return np.concatenate([mean,std,mx,mn],axis=1)  # shape (N, 4*F_sel)

def cv_score(X_feat, label=""):
    cv=StratifiedGroupKFold(6,shuffle=True,random_state=42)
    scores=[]
    for tr,te in cv.split(X_feat,y_enc,g):
        scaler=StandardScaler().fit(X_feat[tr])
        Xt=scaler.transform(X_feat[tr]); Xv=scaler.transform(X_feat[te])
        rf=RandomForestClassifier(
            n_estimators=500,class_weight='balanced',random_state=42)
        rf.fit(Xt,y_enc[tr])
        y_hat=rf.predict(Xv)
        scores.append(f1_score(y_enc[te],y_hat,average='macro'))
    print(f"{label} F1 = {np.mean(scores):.3f} ± {np.std(scores):.3f}")
    return np.mean(scores), rf, scaler

# ───────── 1. Básico vs extendido ─────────
X_basic = make_aggs(X_pad,BASIC)
X_ext   = make_aggs(X_pad,EXTENDED)

f1_b, rf_basic, _ = cv_score(X_basic,"Básico")
f1_e, rf_ext, scaler_ext = cv_score(X_ext,"Extendido")

print("\nΔ F1 =", round(f1_e-f1_b,3))

# ───────── 2. Importancia por permutación ─────────
perm = permutation_importance(
    rf_ext, scaler_ext.transform(X_ext), y_enc,
    n_repeats=30, n_jobs=-1, random_state=42)
imp = pd.Series(perm.importances_mean, index=[f"{stat}_{c}"
       for c in EXTENDED for stat in ['mean','std','max','min']])
imp = imp.sort_values(ascending=False)[:20]

plt.figure(figsize=(8,5))
sns.barplot(x=imp.values, y=imp.index, palette="viridis")
plt.title("Top-20 features más importantes (RF perm-imp)")
plt.xlabel("Importancia media"); plt.ylabel("")
plt.tight_layout(); plt.show()

# ───────── 3. Guardar reporte rápido ─────────
joblib.dump(dict(f1_basic=f1_b,f1_extended=f1_e,importances=imp),
            "rf_feature_ablation_results.pkl")
print("Resultados guardados en rf_feature_ablation_results.pkl")
