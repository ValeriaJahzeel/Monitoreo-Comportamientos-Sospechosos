import streamlit as st
import time
import random
import base64

st.set_page_config(page_title="Simulación de Análisis", layout="wide")

st.title("🎥 Simulación de Detección de Comportamientos Sospechosos")

# === Sección 1: Video cargado ===
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("🔍 Video de vigilancia cargado")
    video_file = open(r'D:\Documentos\Monitoreo-Comportamientos-Sospechosos\dataset\sospechoso\forcejeo\3.mp4', "rb")  # CAMBIA ESTA RUTA
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
        "[INFO] Comparando contra patrones entrenados...",
        "[INFO] Evaluando nivel de riesgo...",
    ]
    
    for i, log in enumerate(logs):
        console_text.markdown(f"```bash\n{log}\n```")
        progress_bar.progress((i+1)/len(logs))
        time.sleep(1.5)

    # Resultado final simulado
    clase_detectada = random.choice([
        ("🟢 COMPORTAMIENTO NORMAL", "green"),
        ("🟡 MERODEO DETECTADO", "orange"),
        ("🔴 INTENTO DE FORZAR ACCESO", "red")
    ])
    
    console_text.markdown(
        f"""```bash
[INFO] Análisis completo.
[RESULTADO] {clase_detectada[0]}
```"""
    )
    st.success(f"✅ Diagnóstico final: {clase_detectada[0]}")

