import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder, StandardScaler

# === CONFIGURACIÓN ===
data_dir = r'D:\Documentos\Monitoreo-Comportamientos-Sospechosos\csv'  # 📁 Carpeta con subcarpetas normal/, merodeo/, forcejeo/
max_frames = 30        # Limita el número de frames por video para acelerar

# === 1. Cargar y unir datos ===
all_data = []
all_labels = []

classes = ['normal', 'merodeo', 'forcejeo']
for label in classes:
    class_dir = os.path.join(data_dir, label)
    for file in os.listdir(class_dir):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(class_dir, file))
            # if 'Frame' in df.columns:
                # df = df[df['Frame'] <= max_frames]  # Limita frames
            df['label'] = label
            all_data.append(df)
            all_labels.extend([label] * len(df))
print(f"📁 Archivos cargados: {len(all_data)}")

# Concatenar todo
df_all = pd.concat(all_data, ignore_index=True)
print(f"🔍 Datos cargados: {df_all.shape[0]} filas, {df_all.shape[1]} columnas")

# === 2. Seleccionar solo columnas numéricas ===
exclude_cols = ['Objeto', 'Objeto_Original']
feature_cols = [c for c in df_all.columns if c not in exclude_cols and df_all[c].dtype in [np.float64, np.int64]]
print(f"🔢 Características numéricas encontradas: {feature_cols}")

# Eliminar NaNs e infinitos
df_all[feature_cols] = df_all[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

# === 3. Codificar labels ===
le = LabelEncoder()
y = le.fit_transform(df_all['label'])

# === 4. Escalar ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_all[feature_cols])

# === 5. Información mutua ===
mi = mutual_info_classif(X_scaled, y, discrete_features=False)
mi_df = pd.DataFrame({'Feature': feature_cols, 'MutualInfo': mi})

# === 6. Importancia con Random Forest ===
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_scaled, y)
rf_importance = rf.feature_importances_
rf_df = pd.DataFrame({'Feature': feature_cols, 'RF_Importance': rf_importance})

# === 7. Combinar resultados ===
result_df = pd.merge(mi_df, rf_df, on='Feature')
result_df = result_df.sort_values(by='RF_Importance', ascending=False)

# === 8. Graficar ===
plt.figure(figsize=(12, 6))
sns.barplot(data=result_df.melt(id_vars='Feature'), x='Feature', y='value', hue='variable')
plt.title('Importancia de características (Random Forest e Información Mutua)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# === 9. Top características sugeridas ===
top_features = result_df.head(10)['Feature'].tolist()
print(f"✅ Top características sugeridas: {top_features}")
