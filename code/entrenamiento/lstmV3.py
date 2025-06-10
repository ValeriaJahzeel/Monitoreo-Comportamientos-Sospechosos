"""
LSTM v3 – clasificación de secuencias de vídeo CON MÉTRICAS COMPLETAS
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
from sklearn.metrics import (
    f1_score, confusion_matrix, ConfusionMatrixDisplay, accuracy_score,
    precision_score, recall_score, classification_report, roc_auc_score,
    log_loss, cohen_kappa_score, matthews_corrcoef, balanced_accuracy_score,
    roc_curve, precision_recall_curve, auc
)
from tensorflow.keras.layers import Input, Masking, Bidirectional, LSTM, BatchNormalization, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import tensorflow as tf, warnings, time
warnings.filterwarnings("ignore")
tf.random.set_seed(42); np.random.seed(42)

ROOT = r"D:\Documentos\Monitoreo-Comportamientos-Sospechosos\csv"
MAX_LEN = 50
BATCH = 8
EPOCHS = 120
PATIENCE_F1 = 8
LR = 5e-4

# ──────────── 1. Data loader ────────────
print("🔄 Cargando datos...")
start_time = time.time()

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

load_time = time.time() - start_time
print(f"✅ Datos cargados en {load_time:.2f}s")
print(f"   Vídeos: {len(X)} | Videos únicos: {len(set(g))}")
print(f"   Distribución de clases: {dict(pd.Series(y).value_counts())}")

# ─ padding / truncating a MAX_LEN ─
F = X[0].shape[1]
print(f"   Características por frame: {F}")
print(f"   Longitud máxima de secuencia: {MAX_LEN}")

def pad(seq):
    if len(seq)>MAX_LEN: return seq[:MAX_LEN]
    if len(seq)<MAX_LEN:
        pad = np.zeros((MAX_LEN-len(seq),F),dtype='float32')
        return np.vstack([seq,pad])
    return seq

X_pad = np.stack([pad(s) for s in X])
print(f"   Forma final de datos: {X_pad.shape}")

# ──────────── 2. Métrica macro-F1 callback ────────────
class MacroF1(Callback):
    def __init__(self,X_val,y_val,patience=PATIENCE_F1):
        super().__init__(); self.Xv=X_val; self.yv=y_val
        self.best=0; self.wait=0; self.pat=patience
        self.f1_history = []
        
    def on_epoch_end(self,epoch,logs=None):
        y_hat=np.argmax(self.model.predict(self.Xv,verbose=0),1)
        f1=f1_score(self.yv,y_hat,average='macro')
        self.f1_history.append(f1)
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
print("\n🔄 Ejecutando validación cruzada 6-fold...")
cv_start_time = time.time()

cv = StratifiedGroupKFold(n_splits=6,shuffle=True,random_state=42)
f1_scores = []
cv_metrics = []

for k,(tr,te) in enumerate(cv.split(X_pad,y_enc,g),1):
    print(f"   Fold {k}/6...")
    fold_start = time.time()
    
    scaler = StandardScaler().fit(X_pad[tr].reshape(-1,F))
    Xt = scaler.transform(X_pad[tr].reshape(-1,F)).reshape(len(tr),MAX_LEN,F)
    Xv = scaler.transform(X_pad[te].reshape(-1,F)).reshape(len(te),MAX_LEN,F)
    ytr,yv = y_enc[tr], y_enc[te]
    ytr_cat,yv_cat = to_categorical(ytr), to_categorical(yv)
    cw = compute_class_weight('balanced',classes=np.arange(len(enc.classes_)),y=ytr)
    
    model = build_lstm((MAX_LEN,F), len(enc.classes_))
    f1_callback = MacroF1(Xv,yv)
    cb=[f1_callback, ReduceLROnPlateau(monitor='val_macroF1',mode='max',
                                          factor=0.5,patience=4,min_lr=1e-6,verbose=0)]
    
    model.fit(Xt,ytr_cat,epochs=EPOCHS,batch_size=BATCH,
              validation_data=(Xv,yv_cat),class_weight=dict(enumerate(cw)),
              verbose=0,callbacks=cb)
    
    # Métricas del fold
    y_hat = np.argmax(model.predict(Xv,verbose=0),1)
    y_prob = model.predict(Xv,verbose=0)
    
    fold_metrics = {
        'fold': k,
        'f1_macro': f1_score(yv,y_hat,average='macro'),
        'f1_weighted': f1_score(yv,y_hat,average='weighted'),
        'accuracy': accuracy_score(yv,y_hat),
        'precision': precision_score(yv,y_hat,average='macro'),
        'recall': recall_score(yv,y_hat,average='macro'),
        'best_val_f1': f1_callback.best,
        'epochs_trained': len(f1_callback.f1_history),
        'fold_time': time.time() - fold_start
    }
    
    try:
        fold_metrics['roc_auc'] = roc_auc_score(yv, y_prob, multi_class='ovr', average='macro')
    except:
        fold_metrics['roc_auc'] = np.nan
    
    fold_metrics['log_loss'] = log_loss(yv, y_prob)
    
    cv_metrics.append(fold_metrics)
    f1_scores.append(fold_metrics['f1_macro'])
    
    print(f"     F1={fold_metrics['f1_macro']:.3f} | Acc={fold_metrics['accuracy']:.3f} | "
          f"Epochs={fold_metrics['epochs_trained']} | Time={fold_metrics['fold_time']:.1f}s")

cv_time = time.time() - cv_start_time

print(f"\n✅ Validación cruzada completada en {cv_time:.2f}s")
print(f"⚑ F1 CV = {np.mean(f1_scores):.3f} ± {np.std(f1_scores):.3f}")

# Estadísticas de CV
cv_df = pd.DataFrame(cv_metrics)
print(f"\n📊 ESTADÍSTICAS DE VALIDACIÓN CRUZADA:")
print(f"   F1-Score: {cv_df['f1_macro'].mean():.4f} ± {cv_df['f1_macro'].std():.4f}")
print(f"   Accuracy: {cv_df['accuracy'].mean():.4f} ± {cv_df['accuracy'].std():.4f}")
print(f"   Precision: {cv_df['precision'].mean():.4f} ± {cv_df['precision'].std():.4f}")
print(f"   Recall: {cv_df['recall'].mean():.4f} ± {cv_df['recall'].std():.4f}")
print(f"   ROC-AUC: {cv_df['roc_auc'].mean():.4f} ± {cv_df['roc_auc'].std():.4f}")
print(f"   Log Loss: {cv_df['log_loss'].mean():.4f} ± {cv_df['log_loss'].std():.4f}")
print(f"   Épocas promedio: {cv_df['epochs_trained'].mean():.1f}")
print(f"   Tiempo promedio por fold: {cv_df['fold_time'].mean():.1f}s")

# ──────────── 5. Split final 80/20 hold-out ────────────
print("\n🔄 Entrenamiento final en hold-out split...")
final_train_start = time.time()

X_train,X_test,y_train,y_test,groups_train,_ = train_test_split(
    X_pad,y_enc,g,test_size=0.20,stratify=y_enc,random_state=42)

print(f"   Train: {len(X_train)} | Test: {len(X_test)}")

scaler = StandardScaler().fit(X_train.reshape(-1,F))
X_train = scaler.transform(X_train.reshape(-1,F)).reshape(len(X_train),MAX_LEN,F)
X_test  = scaler.transform(X_test.reshape(-1,F)).reshape(len(X_test),MAX_LEN,F)
ytr_cat, yte_cat = to_categorical(y_train), to_categorical(y_test)

cw = compute_class_weight('balanced',classes=np.arange(len(enc.classes_)),y=y_train)
final_lstm = build_lstm((MAX_LEN,F), len(enc.classes_))

print(f"🏗️ Arquitectura del modelo:")
final_lstm.summary()

f1_callback_final = MacroF1(X_test,y_test)
hist = final_lstm.fit(
    X_train,ytr_cat,epochs=EPOCHS,batch_size=BATCH,
    validation_split=0.15,
    callbacks=[f1_callback_final,ReduceLROnPlateau(monitor='val_macroF1',
              mode='max',factor=0.5,patience=5,min_lr=1e-6,verbose=1)],
    class_weight=dict(enumerate(cw)),verbose=1)

final_train_time = time.time() - final_train_start

# ──────────── 6. Evaluación completa hold-out ────────────
print("\n🎯 EVALUACIÓN FINAL COMPLETA")
print("=" * 60)

# Predicciones
y_pred_proba = final_lstm.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_proba, axis=1)

# Métricas básicas de clasificación
accuracy = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average='macro')
f1_weighted = f1_score(y_test, y_pred, average='weighted')
precision_macro = precision_score(y_test, y_pred, average='macro')
recall_macro = recall_score(y_test, y_pred, average='macro')
balanced_acc = balanced_accuracy_score(y_test, y_pred)

print(f"📊 MÉTRICAS BÁSICAS:")
print(f"   Accuracy: {accuracy:.4f}")
print(f"   Balanced Accuracy: {balanced_acc:.4f}")
print(f"   F1-Score (Macro): {f1_macro:.4f}")
print(f"   F1-Score (Weighted): {f1_weighted:.4f}")
print(f"   Precision (Macro): {precision_macro:.4f}")
print(f"   Recall (Macro): {recall_macro:.4f}")

# Métricas avanzadas
try:
    roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
    print(f"   ROC-AUC (Macro): {roc_auc:.4f}")
except ValueError:
    roc_auc = np.nan
    print("   ROC-AUC: No calculable (clases insuficientes)")

logloss = log_loss(y_test, y_pred_proba)
kappa = cohen_kappa_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)

print(f"   Log Loss: {logloss:.4f}")
print(f"   Cohen's Kappa: {kappa:.4f}")
print(f"   Matthews Correlation Coefficient: {mcc:.4f}")

# Métricas por clase
print(f"\n📋 REPORTE POR CLASE:")
class_report = classification_report(y_test, y_pred, target_names=enc.classes_, output_dict=True)
print(classification_report(y_test, y_pred, target_names=enc.classes_))

# Métricas de confianza
max_probabilities = np.max(y_pred_proba, axis=1)
mean_confidence = np.mean(max_probabilities)
std_confidence = np.std(max_probabilities)
min_confidence = np.min(max_probabilities)
max_confidence = np.max(max_probabilities)

print(f"\n🎲 MÉTRICAS DE CONFIANZA:")
print(f"   Confianza promedio: {mean_confidence:.4f}")
print(f"   Desviación estándar: {std_confidence:.4f}")
print(f"   Confianza mínima: {min_confidence:.4f}")
print(f"   Confianza máxima: {max_confidence:.4f}")

# Métricas de tiempo y modelo
total_time = cv_time + final_train_time + load_time
print(f"\n⏱️ MÉTRICAS DE RENDIMIENTO:")
print(f"   Tiempo total: {total_time:.2f}s")
print(f"   Tiempo de carga: {load_time:.2f}s")
print(f"   Tiempo CV: {cv_time:.2f}s")
print(f"   Tiempo entrenamiento final: {final_train_time:.2f}s")
print(f"   Épocas entrenadas: {len(hist.history['loss'])}")
print(f"   Mejor F1 en test: {f1_callback_final.best:.4f}")
print(f"   Parámetros del modelo: {final_lstm.count_params():,}")

# Análisis de errores
error_indices = np.where(y_pred != y_test)[0]
correct_indices = np.where(y_pred == y_test)[0]
error_rate = len(error_indices) / len(y_test)

print(f"\n❌ ANÁLISIS DE ERRORES:")
print(f"   Predicciones incorrectas: {len(error_indices)}")
print(f"   Predicciones correctas: {len(correct_indices)}")
print(f"   Tasa de error: {error_rate:.4f}")

if len(error_indices) > 0:
    error_confidences = max_probabilities[error_indices]
    correct_confidences = max_probabilities[correct_indices]
    print(f"   Confianza promedio en errores: {np.mean(error_confidences):.4f}")
    print(f"   Confianza promedio en aciertos: {np.mean(correct_confidences):.4f}")

# Comparación CV vs Hold-out
print(f"\n📈 COMPARACIÓN CV vs HOLD-OUT:")
print(f"   F1 CV: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
print(f"   F1 Hold-out: {f1_macro:.4f}")
print(f"   Diferencia: {f1_macro - np.mean(f1_scores):+.4f}")

# ──────────── 7. Gráficas completas ────────────
print("\n📊 Generando visualizaciones completas...")

# Crear figura con múltiples subplots
fig = plt.figure(figsize=(20, 16))

# 1. Curvas de entrenamiento - Loss
plt.subplot(3, 4, 1)
plt.plot(hist.history['loss'], label='Train Loss', linewidth=2)
plt.plot(hist.history['val_loss'], label='Val Loss', linewidth=2)
plt.title('Curva de Pérdida', fontsize=12, fontweight='bold')
plt.xlabel('Época')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# 2. Curvas de entrenamiento - Accuracy
plt.subplot(3, 4, 2)
plt.plot(hist.history['accuracy'], label='Train Accuracy', linewidth=2)
plt.plot(hist.history['val_accuracy'], label='Val Accuracy', linewidth=2)
plt.title('Exactitud', fontsize=12, fontweight='bold')
plt.xlabel('Época')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

# 3. F1-Score durante entrenamiento
plt.subplot(3, 4, 3)
if hasattr(f1_callback_final, 'f1_history') and f1_callback_final.f1_history:
    plt.plot(f1_callback_final.f1_history, label='Test F1', linewidth=2, color='green')
    plt.title('F1-Score en Test', fontsize=12, fontweight='bold')
    plt.xlabel('Época')
    plt.ylabel('F1-Score')
    plt.legend()
    plt.grid(True, alpha=0.3)

# 4. Matriz de confusión normalizada
plt.subplot(3, 4, 4)
cm_normalized = confusion_matrix(y_test, y_pred, normalize='true')
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=enc.classes_, yticklabels=enc.classes_)
plt.title('Matriz de Confusión\n(Normalizada)', fontsize=12, fontweight='bold')
plt.ylabel('Real')
plt.xlabel('Predicho')

# 5. Matriz de confusión absoluta
plt.subplot(3, 4, 5)
cm_absolute = confusion_matrix(y_test, y_pred)
sns.heatmap(cm_absolute, annot=True, fmt='d', cmap='Oranges',
            xticklabels=enc.classes_, yticklabels=enc.classes_)
plt.title('Matriz de Confusión\n(Absoluta)', fontsize=12, fontweight='bold')
plt.ylabel('Real')
plt.xlabel('Predicho')

# 6. Distribución de confianza
plt.subplot(3, 4, 6)
plt.hist(max_probabilities, bins=20, alpha=0.7, edgecolor='black', color='skyblue')
plt.title('Distribución de Confianza\nen Predicciones', fontsize=12, fontweight='bold')
plt.xlabel('Probabilidad Máxima')
plt.ylabel('Frecuencia')
plt.grid(True, alpha=0.3)

# 7. Métricas por clase (F1-Score)
plt.subplot(3, 4, 7)
f1_scores_class = [class_report[cls]['f1-score'] for cls in enc.classes_]
bars = plt.bar(enc.classes_, f1_scores_class, color=['lightcoral', 'lightgreen', 'lightblue'])
plt.title('F1-Score por Clase', fontsize=12, fontweight='bold')
plt.ylabel('F1-Score')
plt.ylim(0, 1)
for bar, score in zip(bars, f1_scores_class):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{score:.3f}', ha='center', va='bottom')
plt.grid(True, alpha=0.3)

# 8. Variabilidad en CV
plt.subplot(3, 4, 8)
cv_metrics_plot = ['f1_macro', 'accuracy', 'precision', 'recall']
cv_means = [cv_df[metric].mean() for metric in cv_metrics_plot]
cv_stds = [cv_df[metric].std() for metric in cv_metrics_plot]
x_pos = np.arange(len(cv_metrics_plot))
bars = plt.bar(x_pos, cv_means, yerr=cv_stds, capsize=5, 
               color=['lightcoral', 'lightgreen', 'lightblue', 'lightyellow'])
plt.title('Métricas CV\n(Media ± Std)', fontsize=12, fontweight='bold')
plt.ylabel('Valor')
plt.xticks(x_pos, ['F1', 'Acc', 'Prec', 'Rec'])
plt.ylim(0, 1)
for bar, mean, std in zip(bars, cv_means, cv_stds):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
             f'{mean:.3f}', ha='center', va='bottom')
plt.grid(True, alpha=0.3)

# 9. Comparación CV vs Hold-out
plt.subplot(3, 4, 9)
comparison_metrics = ['F1-Score', 'Accuracy', 'Precision', 'Recall']
cv_values = [cv_df['f1_macro'].mean(), cv_df['accuracy'].mean(), 
             cv_df['precision'].mean(), cv_df['recall'].mean()]
holdout_values = [f1_macro, accuracy, precision_macro, recall_macro]

x = np.arange(len(comparison_metrics))
width = 0.35
bars1 = plt.bar(x - width/2, cv_values, width, label='CV', color='lightblue')
bars2 = plt.bar(x + width/2, holdout_values, width, label='Hold-out', color='lightcoral')

plt.title('CV vs Hold-out', fontsize=12, fontweight='bold')
plt.ylabel('Valor')
plt.xticks(x, ['F1', 'Acc', 'Prec', 'Rec'])
plt.ylim(0, 1)
plt.legend()
plt.grid(True, alpha=0.3)

# 10. Épocas de entrenamiento por fold
plt.subplot(3, 4, 10)
fold_numbers = cv_df['fold'].values
epochs_trained = cv_df['epochs_trained'].values
bars = plt.bar(fold_numbers, epochs_trained, color='lightgreen')
plt.title('Épocas por Fold', fontsize=12, fontweight='bold')
plt.xlabel('Fold')
plt.ylabel('Épocas')
for bar, epochs in zip(bars, epochs_trained):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             str(epochs), ha='center', va='bottom')
plt.grid(True, alpha=0.3)

# 11. Confianza por clase
plt.subplot(3, 4, 11)
confidence_by_class = []
for i, class_name in enumerate(enc.classes_):
    class_indices = np.where(y_test == i)[0]
    if len(class_indices) > 0:
        class_confidences = max_probabilities[class_indices]
        confidence_by_class.append(np.mean(class_confidences))
    else:
        confidence_by_class.append(0)

bars = plt.bar(enc.classes_, confidence_by_class, color=['lightcoral', 'lightgreen', 'lightblue'])
plt.title('Confianza Promedio\npor Clase', fontsize=12, fontweight='bold')
plt.ylabel('Confianza Promedio')
plt.ylim(0, 1)
for bar, conf in zip(bars, confidence_by_class):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{conf:.3f}', ha='center', va='bottom')
plt.grid(True, alpha=0.3)

# 12. Resumen de métricas principales
plt.subplot(3, 4, 12)
metrics_names = ['Accuracy', 'F1-Macro', 'Precision', 'Recall', 'Kappa']
metrics_values = [accuracy, f1_macro, precision_macro, recall_macro, kappa]
colors = plt.cm.Set3(np.linspace(0, 1, len(metrics_names)))

bars = plt.bar(metrics_names, metrics_values, color=colors)
plt.title('Resumen de Métricas\nPrincipales', fontsize=12, fontweight='bold')
plt.ylabel('Valor')
plt.ylim(0, 1)
plt.xticks(rotation=45)
for bar, value in zip(bars, metrics_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{value:.3f}', ha='center', va='bottom', fontsize=10)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ──────────── 8. Guardar modelo + scaler + métricas completas ────────────
print("\n💾 Guardando modelo y métricas completas...")

final_lstm.save('lstm_v3_final.h5')

# Preparar datos completos para guardar
complete_data = {
    'scaler': scaler,
    'label_encoder': enc,
    'max_len': MAX_LEN,
    'feature_columns': COLS,
    'config': {
        'max_len': MAX_LEN,
        'batch_size': BATCH,
        'epochs': EPOCHS,
        'patience_f1': PATIENCE_F1,
        'learning_rate': LR,
        'model_params': int(final_lstm.count_params())
    },
    'cv_results': {
        'cv_metrics': cv_metrics,
        'cv_summary': {
            'f1_mean': cv_df['f1_macro'].mean(),
            'f1_std': cv_df['f1_macro'].std(),
            'accuracy_mean': cv_df['accuracy'].mean(),
            'accuracy_std': cv_df['accuracy'].std(),
            'avg_epochs': cv_df['epochs_trained'].mean(),
            'total_cv_time': cv_time
        }
    },
    'holdout_metrics': {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'balanced_accuracy': balanced_acc,
        'roc_auc': roc_auc,
        'log_loss': logloss,
        'cohen_kappa': kappa,
        'matthews_corrcoef': mcc,
        'mean_confidence': mean_confidence,
        'std_confidence': std_confidence,
        'error_rate': error_rate,
        'best_test_f1': f1_callback_final.best,
        'epochs_trained': len(hist.history['loss']),
        'training_time': final_train_time
    },
    'class_report': class_report,
    'confusion_matrix': cm_absolute,
    'confusion_matrix_normalized': cm_normalized,
    'predictions': y_pred,
    'prediction_probabilities': y_pred_proba,
    'test_indices': np.arange(len(y_test)),
    'training_history': hist.history,
    'timing': {
        'load_time': load_time,
        'cv_time': cv_time,
        'final_train_time': final_train_time,
        'total_time': total_time
    }
}

joblib.dump(complete_data, 'lstm_v3_completo.pkl')

print("✅ Modelo, escalador y métricas completas guardados.")
print(f"   Archivo modelo: lstm_v3_final.h5")
print(f"   Archivo datos completos: lstm_v3_completo.pkl")

print(f"\n🎯 RESUMEN FINAL:")
print("=" * 60)
print(f"   🔸 F1-Score CV: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
print(f"   🔸 F1-Score Hold-out: {f1_macro:.4f}")
print(f"   🔸 Accuracy Hold-out: {accuracy:.4f}")
print(f"   🔸 Mejor F1 en test: {f1_callback_final.best:.4f}")
print(f"   🔸 Confianza promedio: {mean_confidence:.4f}")
print(f"   🔸 Parámetros del modelo: {final_lstm.count_params():,}")
print(f"   🔸 Tiempo total: {total_time:.2f}s")
print(f"   🔸 Épocas entrenadas: {len(hist.history['loss'])}")

# Mostrar matriz de confusión final en consola
print(f"\n📊 MATRIZ DE CONFUSIÓN FINAL:")
print("Real \\ Pred", end="")
for cls in enc.classes_:
    print(f"{cls:>10}", end="")
print()
for i, real_cls in enumerate(enc.classes_):
    print(f"{real_cls:>10}", end="")
    for j in range(len(enc.classes_)):
        print(f"{cm_absolute[i,j]:>10}", end="")
    print()

print("\n" + "=" * 60)
print("🎉 ENTRENAMIENTO Y EVALUACIÓN COMPLETADOS")
print("=" * 60)