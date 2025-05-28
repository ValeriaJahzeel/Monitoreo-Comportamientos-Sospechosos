import streamlit as st
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Comportamientos en Video", layout="wide")

# === Título de la página ===
st.title("🔍 Análisis de Comportamientos Sospechosos en Video")
st.markdown("Explora cómo se analizan comportamientos como merodeo, forcejeo e interacción normal a través de datos extraídos de videos de vigilancia.")

# === Ruta a las carpetas de datos ===
RUTA_BASE = r'D:\Documentos\Monitoreo-Comportamientos-Sospechosos\csv' # CAMBIA ESTA RUTA
CARPETAS = {
    "normal": os.path.join(RUTA_BASE, "normal"),
    "merodeo": os.path.join(RUTA_BASE, "merodeo"),
    "forcejeo": os.path.join(RUTA_BASE, "forcejeo")
}

# === Carga de archivos ===
@st.cache_data
def cargar_datos():
    dataframes = []
    for clase, ruta in CARPETAS.items():
        for archivo in os.listdir(ruta):
            if archivo.endswith(".csv"):
                try:
                    df = pd.read_csv(os.path.join(ruta, archivo))
                    df["clase"] = clase
                    df["archivo"] = archivo
                    dataframes.append(df)
                except Exception as e:
                    st.warning(f"Error en {archivo}: {e}")
    return pd.concat(dataframes, ignore_index=True)

df_all = cargar_datos()

# === Sección: Estadísticas ===
st.subheader("📊 Distribución de muestras por clase")
conteo = df_all["clase"].value_counts().reset_index()
conteo.columns = ["Clase", "Número de muestras"]
st.dataframe(conteo)

fig1, ax1 = plt.subplots()
sns.countplot(data=df_all, x="clase", palette="Set2", ax=ax1)
ax1.set_title("Distribución de clases")
st.pyplot(fig1)

# === Sección: Exploración numérica ===
columnas_numericas = df_all.select_dtypes(include="number").columns.tolist()
if columnas_numericas:
    st.subheader("📈 Comparación por característica")
    col_seleccionada = st.selectbox("Selecciona una característica para comparar:", columnas_numericas)

    fig2, ax2 = plt.subplots()
    sns.boxplot(data=df_all, x="clase", y=col_seleccionada, palette="Set3", ax=ax2)
    ax2.set_title(f"Distribución de '{col_seleccionada}' por clase")
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots()
    sns.histplot(data=df_all, x=col_seleccionada, hue="clase", kde=True, element="step", ax=ax3)
    ax3.set_title(f"Histograma de '{col_seleccionada}' por clase")
    st.pyplot(fig3)

# === Sección: Muestra aleatoria ===
st.subheader("🔍 Muestra aleatoria de datos")
st.dataframe(df_all.sample(5))

st.markdown("---")
st.markdown("👁️‍🗨️ Esta aplicación muestra cómo la IA puede aprender a distinguir comportamientos humanos a partir de datos extraídos de videos.")

