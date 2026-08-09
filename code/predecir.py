"""
Demo de punta a punta: clasifica un video nuevo como normal / merodeo /
forcejeo. Corre el pipeline completo (detección + tracking + extracción de
características + predicción) sobre un solo video.

Uso:
    python code/predecir.py --video ruta/al/video.mp4
    python code/predecir.py --video ruta/al/video.mp4 --modelo random_forest
    python code/predecir.py --video ruta/al/video.mp4 --guardar_video salida.mp4

Modelos disponibles (ver README para la comparación con validación cruzada):
    gradient_boosting (default, ~72% accuracy), random_forest (~70%),
    mlp (~58%), lstm (~60%)
"""

import argparse
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "entrenamiento"))

from objectDetection import ObjectDetector


def extraer_caracteristicas(video_path, csv_path, guardar_video=None):
    detector = ObjectDetector()
    detector.procesar_video(
        video_path, csv_path, mostrar=False, usar_threads=False, guardar_video=guardar_video
    )


def predecir(video_path, modelo_nombre, guardar_video=None, csv_path=None):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"No existe el video {video_path}")

    csv_temporal = csv_path is None
    if csv_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        csv_path = tmp.name
        tmp.close()

    print(f"Analizando {video_path}...")
    extraer_caracteristicas(video_path, csv_path, guardar_video=guardar_video)

    try:
        if modelo_nombre in ("random_forest", "gradient_boosting"):
            from modelo_arboles import predecir_video
            modelo_path = f"best_model_{modelo_nombre}.pkl"
            if not os.path.exists(modelo_path):
                raise FileNotFoundError(
                    f"No se encontró {modelo_path}. Entrena primero con "
                    f"code/entrenamiento/modelo_arboles.py"
                )
            resultado = predecir_video(modelo_path, csv_path)
        else:
            import model as model_module
            modelo_path = f"best_model_{modelo_nombre}_78videos.pth"
            if not os.path.exists(modelo_path):
                modelo_path = f"best_model_{modelo_nombre}.pth"
            if not os.path.exists(modelo_path):
                raise FileNotFoundError(
                    f"No se encontró un checkpoint de {modelo_nombre}. Entrena primero con "
                    f"code/model.py --mode train --model_type {modelo_nombre}"
                )
            modelo, checkpoint = model_module.load_best_model(modelo_path)
            scaler = model_module.load_scaler_from_checkpoint(checkpoint)
            class_names = checkpoint.get("class_names", ["normal", "merodeo", "forcejeo"])
            resultado = model_module.predict_video(modelo, csv_path, scaler=scaler, class_names=class_names)
    finally:
        if csv_temporal and os.path.exists(csv_path):
            os.remove(csv_path)

    return resultado


def imprimir_resultado(resultado, modelo_nombre):
    print(f"\n{'='*50}")
    print(f"Modelo usado: {modelo_nombre}")
    print(f"Predicción: {resultado['predicted_label'].upper()}")
    print(f"{'='*50}")
    for nombre, prob in sorted(resultado["probabilities"].items(), key=lambda x: -x[1]):
        barra = "█" * int(prob / 2.5)
        print(f"  {nombre:12s} {prob:5.1f}%  {barra}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clasificar un video como normal/merodeo/forcejeo")
    parser.add_argument("--video", type=str, required=True, help="Ruta al video a analizar")
    parser.add_argument("--modelo", type=str, default="gradient_boosting",
                         choices=["gradient_boosting", "random_forest", "mlp", "lstm"],
                         help="Modelo a usar para la predicción")
    parser.add_argument("--guardar_video", type=str, default=None,
                         help="Ruta opcional para guardar el video anotado (cajas, IDs, trayectorias)")
    parser.add_argument("--csv", type=str, default=None,
                         help="Ruta opcional para guardar el CSV de características (por defecto, temporal)")
    args = parser.parse_args()

    resultado = predecir(args.video, args.modelo, guardar_video=args.guardar_video, csv_path=args.csv)
    imprimir_resultado(resultado, args.modelo)
