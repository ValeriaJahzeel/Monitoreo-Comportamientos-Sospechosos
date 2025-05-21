import pandas as pd
import matplotlib.pyplot as plt

# === Datos cargados manualmente (puedes cambiar esto por lectura CSV si los tienes en archivo) ===
data = {
    'Variable': ['Centroide_X', 'Centroide_Y', 'Desplazamiento', 'Velocidad', 'Aceleracion', 'Direccion',
                 'Densidad', 'Linealidad', 'Circularidad', 'Zigzag', 'Es_Ciclico', 'Frecuencia_Ciclo',
                 'Amplitud_Ciclo', 'Area_Trayectoria', 'En_Interaccion'],
    'mean': [317.06, 141.41, 117.37, 2934.36, 27.55, 0.89, 0.0091, 0.018, 0.308, 0.648, 0.174, 0.061, 7.36, 61033.13, 0.0104],
    'std': [134.19, 66.34, 110.05, 2751.29, 70679.91, 104.7, 0.0053, 0.0898, 0.699, 0.0967, 0.379, 0.153, 23.28, 40787.8, 0.1013],
    'std/mean': [0.42, 0.47, 0.94, 0.94, 2565.80, 117.51, 0.59, 4.98, 2.27, 0.15, 2.18, 2.52, 3.16, 0.67, 9.77]
}

df = pd.DataFrame(data)

# === 1. Gráfica de medias por variable ===
plt.figure(figsize=(12, 6))
plt.bar(df['Variable'], df['mean'], color='skyblue')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Media (mean)')
plt.title('Media de cada variable')
plt.tight_layout()
plt.show()

# === 2. Gráfica del coeficiente de variación (std/mean) ===
plt.figure(figsize=(12, 6))
plt.bar(df['Variable'], df['std/mean'], color='salmon')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Coeficiente de variación (std/mean)')
plt.title('Variabilidad relativa por variable')
plt.tight_layout()
plt.show()

# === 3. (Opcional) Gráfica comparativa entre std y mean ===
plt.figure(figsize=(12, 6))
bar_width = 0.4
index = range(len(df))

plt.bar(index, df['mean'], bar_width, label='Media', color='lightgreen')
plt.bar([i + bar_width for i in index], df['std'], bar_width, label='Desviación Estándar', color='orange')
plt.xticks([i + bar_width / 2 for i in index], df['Variable'], rotation=45, ha='right')
plt.ylabel('Valor')
plt.title('Media vs Desviación estándar por variable')
plt.legend()
plt.tight_layout()
plt.show()
