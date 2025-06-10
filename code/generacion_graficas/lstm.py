import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

# --- 1. Generar datos de ejemplo (ajusta esto con tus datos reales) ---
np.random.seed(42) # Para reproducibilidad

# Datos para Pérdida y Precisión durante entrenamiento
epochs = np.arange(0, 41)

# --- Datos de Pérdida (Loss) ---
# Train Loss: Empieza alto, baja, luego tiene algunas fluctuaciones y picos.
train_loss = np.zeros(len(epochs))
train_loss[:10] = np.linspace(1.2, 0.7, 10) + np.random.normal(0, 0.05, 10) # Baja inicial
train_loss[10:20] = np.linspace(0.7, 0.5, 10) + np.random.normal(0, 0.03, 10) # Sigue bajando
train_loss[20:25] = np.linspace(0.5, 0.65, 5) + np.random.normal(0, 0.08, 5) # Pequeño pico
train_loss[25:30] = np.linspace(0.65, 0.4, 5) + np.random.normal(0, 0.05, 5) # Baja de nuevo
train_loss[30:] = np.linspace(0.4, 0.5, len(epochs)-30) + np.random.normal(0, 0.05, len(epochs)-30) # Sube y fluctúa al final
train_loss = np.clip(train_loss, 0.3, 1.3) # Asegurarse de que los valores sean razonables

# Val Loss: Comportamiento más errático, con valles y picos pronunciados
val_loss = np.zeros(len(epochs))
val_loss[:5] = np.linspace(1.1, 0.9, 5) + np.random.normal(0, 0.1, 5) # Baja inicial
val_loss[5:15] = np.linspace(0.9, 0.7, 10) + np.random.normal(0, 0.08, 10) # Sigue bajando
val_loss[15:20] = np.linspace(0.7, 0.6, 5) + np.random.normal(0, 0.05, 5) # Baja más
val_loss[20:25] = np.linspace(0.6, 0.75, 5) + np.random.normal(0, 0.1, 5) # Pico ascendente
val_loss[25:30] = np.linspace(0.75, 0.6, 5) + np.random.normal(0, 0.08, 5) # Baja
val_loss[30:35] = np.linspace(0.6, 0.9, 5) + np.random.normal(0, 0.1, 5) # Gran pico
val_loss[35:] = np.linspace(0.9, 0.8, len(epochs)-35) + np.random.normal(0, 0.05, len(epochs)-35) # Baja final
val_loss = np.clip(val_loss, 0.4, 1.2) # Asegurarse de que los valores sean razonables

# --- Datos de Precisión (Accuracy) ---
# Train Accuracy: Sube constantemente y se estabiliza alto.
train_accuracy = np.zeros(len(epochs))
train_accuracy[:10] = np.linspace(0.8, 0.9, 10) + np.random.normal(0, 0.02, 10)
train_accuracy[10:20] = np.linspace(0.9, 0.85, 10) + np.random.normal(0, 0.03, 10) # Ligeras fluctuaciones
train_accuracy[20:30] = np.linspace(0.85, 0.9, 10) + np.random.normal(0, 0.02, 10)
train_accuracy[30:] = np.linspace(0.9, 0.87, len(epochs)-30) + np.random.normal(0, 0.03, len(epochs)-30)
train_accuracy = np.clip(train_accuracy, 0.75, 0.95) # Asegurarse de que los valores sean razonables

# Val Accuracy: Comienza bajo, sube con fluctuaciones y se mantiene irregular en un rango medio-alto.
val_accuracy = np.zeros(len(epochs))
val_accuracy[:5] = np.linspace(0.2, 0.7, 5) + np.random.normal(0, 0.05, 5) # Subida inicial rápida
val_accuracy[5:15] = np.linspace(0.7, 0.6, 10) + np.random.normal(0, 0.05, 10) # Baja y fluctúa
val_accuracy[15:20] = np.linspace(0.6, 0.8, 5) + np.random.normal(0, 0.08, 5) # Sube con pico
val_accuracy[20:25] = np.linspace(0.8, 0.7, 5) + np.random.normal(0, 0.05, 5) # Baja
val_accuracy[25:30] = np.linspace(0.7, 0.85, 5) + np.random.normal(0, 0.08, 5) # Sube otro pico
val_accuracy[30:35] = np.linspace(0.85, 0.7, 5) + np.random.normal(0, 0.05, 5) # Baja
val_accuracy[35:] = np.linspace(0.7, 0.75, len(epochs)-35) + np.random.normal(0, 0.03, len(epochs)-35) # Se estabiliza con fluctuaciones
val_accuracy = np.clip(val_accuracy, 0.2, 0.9) # Asegurarse de que los valores sean razonables


# Datos para Matriz de Confusión (ejemplo, ajusta con tus verdaderos y predichos)
y_true = np.array(['forcejeo', 'forcejeo', 'forcejeo', 'merodeo', 'merodeo', 'merodeo', 'merodeo', 'normal', 'normal', 'normal', 'normal'])
y_pred = np.array(['forcejeo', 'forcejeo', 'merodeo', 'forcejeo', 'merodeo', 'merodeo', 'merodeo', 'forcejeo', 'normal', 'normal', 'normal'])
labels = ['forcejeo', 'merodeo', 'normal']
cm = confusion_matrix(y_true, y_pred, labels=labels)

# Normalizar la matriz de confusión
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Datos para Distribución de Confianza en Predicciones
max_probabilities = np.concatenate([
    np.random.normal(loc=0.45, scale=0.05, size=30), # Clúster inferior
    np.random.normal(loc=0.7, scale=0.08, size=40),  # Clúster medio
    np.random.normal(loc=0.95, scale=0.03, size=20)  # Clúster superior
])
max_probabilities = np.clip(max_probabilities, 0.0, 1.0) # Asegurarse de que estén entre 0 y 1

# --- 2. Crear las gráficas en ventanas separadas ---

# Gráfica 1: Pérdida durante entrenamiento
plt.figure("Pérdida durante entrenamiento", figsize=(8, 6)) # Nombre de la ventana y tamaño
plt.plot(epochs, train_loss, label='Train Loss')
plt.plot(epochs, val_loss, label='Val Loss')
plt.title('Pérdida durante entrenamiento')
plt.xlabel('Época')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.ylim(0.3, 1.3) # Ajustar límites Y para mejor visualización
plt.xticks(np.arange(0, len(epochs), 10)) # Marcas en el eje X cada 10 épocas


# Gráfica 2: Precisión durante entrenamiento
plt.figure("Precisión durante entrenamiento", figsize=(8, 6)) # Nombre de la ventana y tamaño
plt.plot(epochs, train_accuracy, label='Train Accuracy')
plt.plot(epochs, val_accuracy, label='Val Accuracy')
plt.title('Precisión durante entrenamiento')
plt.xlabel('Época')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()
plt.ylim(0.2, 0.95) # Ajustar límites Y para mejor visualización
plt.xticks(np.arange(0, len(epochs), 10)) # Marcas en el eje X cada 10 épocas


# Gráfica 3: Matriz de Confusión Normalizada
plt.figure("Matriz de Confusión Normalizada", figsize=(7, 6)) # Nombre de la ventana y tamaño
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Matriz de Confusión Normalizada')
plt.xlabel('Predicho')
plt.ylabel('Real')

# Gráfica 4: Distribución de Confianza en Predicciones
plt.figure("Distribución de Confianza en Predicciones", figsize=(8, 6)) # Nombre de la ventana y tamaño
plt.hist(max_probabilities, bins=10, edgecolor='black', alpha=0.7)
plt.title('Distribución de Confianza en Predicciones')
plt.xlabel('Probabilidad Máxima')
plt.ylabel('Frecuencia')
plt.grid(True, linestyle=':', alpha=0.6)

# Mostrar todas las figuras
plt.show()