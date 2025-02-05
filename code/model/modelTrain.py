import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Directorio de los datos
dataset_dir = "normalized_dataset"

# Parámetros
img_size = (96, 96)  # Tamaño de las imágenes
batch_size = 32  # Tamaño del batch
epochs = 10  # Número de épocas

# Generador de datos
datagen = ImageDataGenerator(
    rescale=1./255,  # Normalización de los píxeles a [0, 1]
    validation_split=0.2  # 80% entrenamiento, 20% validación
)

# Datos de entrenamiento
train_data = datagen.flow_from_directory(
    dataset_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="training"
)

# Datos de validación
val_data = datagen.flow_from_directory(
    dataset_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation"
)

# Obtener las clases
class_names = list(train_data.class_indices.keys())
print(f"Clases detectadas: {class_names}")

# Modelo de red neuronal convolucional (CNN)
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(96, 96, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(len(class_names), activation="softmax")  # Una salida por clase
])

# Compilar el modelo
model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# Entrenar el modelo
history = model.fit(
    train_data,
    epochs=epochs,
    validation_data=val_data
)

# Guardar el modelo
model.save("xd/modelo.h5")
print("Modelo guardado como 'modelo.h5'.")

# Graficar las métricas de entrenamiento
plt.plot(history.history["accuracy"], label="Precisión entrenamiento")
plt.plot(history.history["val_accuracy"], label="Precisión validación")
plt.xlabel("Épocas")
plt.ylabel("Precisión")
plt.legend(loc="lower right")
plt.title("Precisión del modelo")
plt.show()