import tensorflow as tf
import cv2
import numpy as np

# Cargar el modelo
model = tf.keras.models.load_model("modelo.h5")
class_names = ["derecha", "izquierda", "stop"]  # Ajusta según tus clases

def classify_image(img_path):
    # Cargar la imagen
    img = cv2.imread(img_path)
    img = cv2.resize(img, (96, 96)) / 255.0
    img = np.expand_dims(img, axis=0)

    # Clasificar
    predictions = model.predict(img)
    class_idx = np.argmax(predictions)
    confidence = predictions[0][class_idx]
    return class_names[class_idx], confidence

# Clasificar una imagen de prueba
img_path = "xd/imagenes/alto.jpg"
label, confidence = classify_image(img_path)
print(f"Señal detectada: {label}, Confianza: {confidence*100:.2f}%")