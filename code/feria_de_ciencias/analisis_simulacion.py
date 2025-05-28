import streamlit as st
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import time
import random

st.set_page_config(page_title="Análisis de Comportamientos", layout="wide")

st.title("🎥 Detección Simulada de Comportamientos Sospechosos")
st.markdown("Esta aplicación simula el análisis de comportamientos en videos de vigilancia. Carga datos y simula un análisis estilo 'película' para hacerlo más interesante para estudiantes.")

# === Rutas de carpetas ===
RUTA_BASE = r'D:\Documentos\Monitoreo-Comportamientos-Sospechosos\csv'  # <-- CAMBIA ESTA RUTA
CARPETAS = {
    "normal": os.path.join(RUTA_BASE, "normal"),
    "merodeo": os.path.join(RUTA_BASE, "merodeo"),
    "forcejeo": os.path.join(RUTA_BASE, "forcejeo")
}

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

# === Carga de datos ===
st.header("📁 1. Exploración de Datos")
df_all = cargar_datos()
st.success("Datos cargados correctamente.")

st.subheader("📊 Distribución de comportamientos")
conteo = df_all["clase"].value_counts().reset_index()
conteo.columns = ["Comportamiento", "Número de muestras"]
st.dataframe(conteo)

fig1, ax1 = plt.subplots()
sns.countplot(data=df_all, x="clase", palette="Set2", ax=ax1)
ax1.set_title("Número de muestras por clase")
st.pyplot(fig1)

columnas_numericas = df_all.select_dtypes(include="number").columns.tolist()
if columnas_numericas:
    st.subheader("📈 Compara características entre clases")
    caracteristica = st.selectbox("Selecciona una característica para visualizar:", columnas_numericas)

    fig2, ax2 = plt.subplots()
    sns.boxplot(data=df_all, x="clase", y=caracteristica, palette="Set3", ax=ax2)
    ax2.set_title(f"Distribución de '{caracteristica}' por clase")
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots()
    sns.histplot(data=df_all, x=caracteristica, hue="clase", kde=True, element="step", ax=ax3)
    ax3.set_title(f"Histograma de '{caracteristica}'")
    st.pyplot(fig3)

# === Simulación estilo película ===
st.header("🎬 2. Simulación de Análisis Tipo Película")
st.markdown("Haz clic en el siguiente botón para seleccionar un video aleatorio y simular el análisis:")

if st.button("🚀 Probar algoritmo"):
    RUTA_VIDEOS = r'D:\Documentos\Monitoreo-Comportamientos-Sospechosos\dataset'  # <-- CAMBIA ESTA RUTA
    VIDEOS = {
        "normal": os.path.join(RUTA_VIDEOS, "normal"),
        "merodeo": os.path.join(RUTA_VIDEOS, "merodeo"),
        "forcejeo": os.path.join(RUTA_VIDEOS, "forcejeo")
    }

    # Selección aleatoria de clase y video
    clase_seleccionada = random.choice(list(VIDEOS.keys()))
    carpeta_video = VIDEOS[clase_seleccionada]
    videos_disponibles = [v for v in os.listdir(carpeta_video) if v.endswith(".mp4")]
    video_elegido = random.choice(videos_disponibles)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader(f"📹 Video seleccionado aleatoriamente: {video_elegido}")
        with open(os.path.join(carpeta_video, video_elegido), "rb") as video_file:
            video_bytes = video_file.read()
            st.video(video_bytes)

    with col2:
        st.subheader("🖥️ Consola de Análisis")

        console_text = st.empty()
        progress_bar = st.progress(0)

        logs = [
            "[INFO] Inicializando análisis...",
            "[INFO] Cargando datos de trayectoria...",
            "[INFO] Calculando velocidad y forma de movimiento...",
            "[INFO] Comparando contra patrones aprendidos...",
            "[INFO] Evaluando nivel de riesgo...",
        ]

        for i, log in enumerate(logs):
            console_text.markdown(f"```bash\n{log}\n```")
            progress_bar.progress((i + 1) / len(logs))
            time.sleep(1.3)

        clases_humanas = {
            "normal": "🟢 COMPORTAMIENTO NORMAL",
            "merodeo": "🟡 MERODEO DETECTADO",
            "forcejeo": "🔴 INTENTO DE FORZAR ACCESO"
        }

        console_text.markdown(f"""```bash
[INFO] Análisis completo.
[RESULTADO] {clases_humanas[clase_seleccionada]}
```""")
        st.success(f"✅ Diagnóstico final: {clases_humanas[clase_seleccionada]}")

st.markdown("---")
st.markdown("🎓 Este demo fue creado para mostrar cómo se pueden detectar comportamientos en cámaras de seguridad usando datos y modelos de inteligencia artificial. ¡Explora, experimenta y aprende!")
