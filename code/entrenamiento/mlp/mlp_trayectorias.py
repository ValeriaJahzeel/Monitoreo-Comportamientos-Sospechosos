#!/usr/bin/env python
# mlp_trayectorias_final_with_metrics.py
import numpy as np, pandas as pd, joblib, matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.metrics import (
    f1_score, confusion_matrix, ConfusionMatrixDisplay, accuracy_score,
    precision_score, recall_score, classification_report, roc_auc_score,
    log_loss, cohen_kappa_score, matthews_corrcoef, balanced_accuracy_score,
    roc_curve, precision_recall_curve, auc
)
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, BatchNormalization, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import seaborn as sns
import time

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

print("🔄 Cargando datos...")
start_time = time.time()

X, y, g = [], [], []
for cls in ['normal','merodeo','forcejeo']:
    for csv in Path(ROOT, cls).glob('*.csv'):
        for seq, lbl, vid in csv2seq(csv):
            X.append(seq); y.append(lbl); g.append(vid)

X = np.stack(X); y = np.array(y); g = np.array(g)
enc = LabelEncoder(); y_enc = enc.fit_transform(y)

load_time = time.time() - start_time
print(f"✅ Datos cargados en {load_time:.2f}s")
print(f"   Secuencias: {X.shape} | Videos únicos: {len(set(g))}")
print(f"   Distribución de clases: {dict(pd.Series(y).value_counts())}")

# ---------------- agregados ----------------
print("🔄 Generando características agregadas...")
X_agg = np.concatenate([X.mean(1), X.std(1), X.max(1), X.min(1)], axis=1)
print(f"   Características agregadas: {X_agg.shape[1]}")

# ---------------- train / hold-out split ----------------
print("🔄 Dividiendo datos...")
X_train, X_test, y_train, y_test = train_test_split(
    X_agg, y_enc, test_size=0.20, stratify=y_enc, random_state=42)

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train); X_test = scaler.transform(X_test)
n_cls = len(enc.classes_)

print(f"   Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

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

print("🏗️ Construyendo modelo MLP...")
ytr_cat = to_categorical(y_train, n_cls); yte_cat = to_categorical(y_test, n_cls)
cw = compute_class_weight(class_weight='balanced',
                          classes=np.arange(n_cls), y=y_train)

mlp = build_mlp(X_train.shape[1], n_cls, CFG)
mlp.summary()

print("🏋️ Entrenando modelo...")
train_start = time.time()

hist = mlp.fit(
    X_train, ytr_cat,
    validation_split=0.15,
    epochs=CFG['epochs'],
    batch_size=CFG['batch'],
    class_weight=dict(enumerate(cw)),
    callbacks=[EarlyStopping(patience=CFG['patience'],
                             restore_best_weights=True)],
    verbose=1)

train_time = time.time() - train_start
print(f"✅ Entrenamiento completado en {train_time:.2f}s")

# ---------------- evaluación completa ----------------
print("\n🎯 EVALUACIÓN COMPLETA")
print("=" * 50)

# Predicciones
y_pred_proba = mlp.predict(X_test, verbose=0)
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

# Métricas de tiempo
total_predictions = len(y_test)
inference_time_per_sample = train_time / len(X_train) * 1000  # ms por muestra durante entrenamiento
print(f"\n⏱️ MÉTRICAS DE RENDIMIENTO:")
print(f"   Tiempo de carga de datos: {load_time:.2f}s")
print(f"   Tiempo de entrenamiento: {train_time:.2f}s")
print(f"   Predicciones totales: {total_predictions}")
print(f"   Parámetros del modelo: {mlp.count_params():,}")

# Análisis de errores
print(f"\n❌ ANÁLISIS DE ERRORES:")
error_indices = np.where(y_pred != y_test)[0]
correct_indices = np.where(y_pred == y_test)[0]
error_rate = len(error_indices) / len(y_test)

print(f"   Predicciones incorrectas: {len(error_indices)}")
print(f"   Predicciones correctas: {len(correct_indices)}")
print(f"   Tasa de error: {error_rate:.4f}")

if len(error_indices) > 0:
    error_confidences = max_probabilities[error_indices]
    correct_confidences = max_probabilities[correct_indices]
    print(f"   Confianza promedio en errores: {np.mean(error_confidences):.4f}")
    print(f"   Confianza promedio en aciertos: {np.mean(correct_confidences):.4f}")

# ---------------- gráficas completas ----------------
print("\n📊 Generando visualizaciones...")

# Crear subplots para todas las gráficas
fig = plt.figure(figsize=(20, 15))

# 1. Curva de pérdida
plt.subplot(3, 4, 1)
plt.plot(hist.history['loss'], label='Train Loss', linewidth=2)
plt.plot(hist.history['val_loss'], label='Val Loss', linewidth=2)
plt.title('Curva de Pérdida', fontsize=12, fontweight='bold')
plt.xlabel('Época')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# 2. Exactitud
plt.subplot(3, 4, 2)
plt.plot(hist.history['accuracy'], label='Train Accuracy', linewidth=2)
plt.plot(hist.history['val_accuracy'], label='Val Accuracy', linewidth=2)
plt.title('Exactitud', fontsize=12, fontweight='bold')
plt.xlabel('Época')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

# 3. Matriz de confusión normalizada
plt.subplot(3, 4, 3)
cm_normalized = confusion_matrix(y_test, y_pred, normalize='true')
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=enc.classes_, yticklabels=enc.classes_)
plt.title('Matriz de Confusión\n(Normalizada)', fontsize=12, fontweight='bold')
plt.ylabel('Real')
plt.xlabel('Predicho')

# 4. Matriz de confusión absoluta
plt.subplot(3, 4, 4)
cm_absolute = confusion_matrix(y_test, y_pred)
sns.heatmap(cm_absolute, annot=True, fmt='d', cmap='Oranges',
            xticklabels=enc.classes_, yticklabels=enc.classes_)
plt.title('Matriz de Confusión\n(Absoluta)', fontsize=12, fontweight='bold')
plt.ylabel('Real')
plt.xlabel('Predicho')

# 5. Distribución de confianza
plt.subplot(3, 4, 5)
plt.hist(max_probabilities, bins=20, alpha=0.7, edgecolor='black', color='skyblue')
plt.title('Distribución de Confianza\nen Predicciones', fontsize=12, fontweight='bold')
plt.xlabel('Probabilidad Máxima')
plt.ylabel('Frecuencia')
plt.grid(True, alpha=0.3)

# 6. Métricas por clase (F1-Score)
plt.subplot(3, 4, 6)
f1_scores = [class_report[cls]['f1-score'] for cls in enc.classes_]
bars = plt.bar(enc.classes_, f1_scores, color=['lightcoral', 'lightgreen', 'lightblue'])
plt.title('F1-Score por Clase', fontsize=12, fontweight='bold')
plt.ylabel('F1-Score')
plt.ylim(0, 1)
for bar, score in zip(bars, f1_scores):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{score:.3f}', ha='center', va='bottom')
plt.grid(True, alpha=0.3)

# 7. Precisión por clase
plt.subplot(3, 4, 7)
precision_scores = [class_report[cls]['precision'] for cls in enc.classes_]
bars = plt.bar(enc.classes_, precision_scores, color=['salmon', 'lightgreen', 'lightskyblue'])
plt.title('Precisión por Clase', fontsize=12, fontweight='bold')
plt.ylabel('Precisión')
plt.ylim(0, 1)
for bar, score in zip(bars, precision_scores):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{score:.3f}', ha='center', va='bottom')
plt.grid(True, alpha=0.3)

# 8. Recall por clase
plt.subplot(3, 4, 8)
recall_scores = [class_report[cls]['recall'] for cls in enc.classes_]
bars = plt.bar(enc.classes_, recall_scores, color=['lightpink', 'lightgreen', 'powderblue'])
plt.title('Recall por Clase', fontsize=12, fontweight='bold')
plt.ylabel('Recall')
plt.ylim(0, 1)
for bar, score in zip(bars, recall_scores):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{score:.3f}', ha='center', va='bottom')
plt.grid(True, alpha=0.3)

# 9. Distribución de clases en test
plt.subplot(3, 4, 9)
unique, counts = np.unique(y_test, return_counts=True)
class_names_test = [enc.classes_[i] for i in unique]
colors = ['lightcoral', 'lightgreen', 'lightblue']
bars = plt.bar(class_names_test, counts, color=colors[:len(unique)])
plt.title('Distribución de Clases\nen Test', fontsize=12, fontweight='bold')
plt.ylabel('Cantidad')
for bar, count in zip(bars, counts):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             str(count), ha='center', va='bottom')
plt.grid(True, alpha=0.3)

# 10. Confianza por clase
plt.subplot(3, 4, 10)
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

# 11. Evolución del learning rate (si está disponible)
plt.subplot(3, 4, 11)
epochs = range(1, len(hist.history['loss']) + 1)
plt.plot(epochs, hist.history['loss'], 'b-', label='Training Loss', linewidth=2)
plt.plot(epochs, hist.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
plt.fill_between(epochs, hist.history['loss'], alpha=0.3, color='blue')
plt.fill_between(epochs, hist.history['val_loss'], alpha=0.3, color='red')
plt.title('Evolución Detallada\nde la Pérdida', fontsize=12, fontweight='bold')
plt.xlabel('Época')
plt.ylabel('Loss')
plt.legend()
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

# ---------------- guardado con métricas ----------------
print("\n💾 Guardando modelo y métricas...")

# Guardar modelo
mlp.save('mlp_trayectorias_final.h5')

# Guardar preprocessors y métricas
model_data = {
    'scaler': scaler,
    'label_encoder': enc,
    'metrics': {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'balanced_accuracy': balanced_acc,
        'log_loss': logloss,
        'cohen_kappa': kappa,
        'matthews_corrcoef': mcc,
        'mean_confidence': mean_confidence,
        'std_confidence': std_confidence,
        'error_rate': error_rate,
        'training_time': train_time,
        'load_time': load_time,
        'model_parameters': int(mlp.count_params())
    },
    'class_report': class_report,
    'confusion_matrix': cm_absolute,
    'confusion_matrix_normalized': cm_normalized,
    'predictions': y_pred,
    'prediction_probabilities': y_pred_proba,
    'test_indices': np.arange(len(y_test)),
    'config': CFG
}

joblib.dump(model_data, 'mlp_trayectorias_completo.pkl')

print("✅ Modelo, escalador y métricas guardados.")
print(f"   Archivo modelo: mlp_trayectorias_final.h5")
print(f"   Archivo datos: mlp_trayectorias_completo.pkl")
print(f"\n🎯 RESUMEN FINAL:")
print(f"   Mejor métrica (F1-Macro): {f1_macro:.4f}")
print(f"   Accuracy: {accuracy:.4f}")
print(f"   Confianza promedio: {mean_confidence:.4f}")
print("=" * 50)