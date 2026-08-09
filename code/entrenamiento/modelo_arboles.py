"""
Modelo de comparación basado en árboles (Random Forest / Gradient Boosting)
sobre estadísticas agregadas por video.

A diferencia de model.py (LSTM/MLP), que consumen la secuencia completa de
frames de cada video, aquí cada video se resume en UN solo vector de
características (media, desviación, máximo, etc. de cada columna numérica a
lo largo de todos sus frames). Con un dataset chico (decenas de videos) un
ensamble de árboles suele generalizar mejor que una red profunda, entrena en
segundos y sus feature_importances_ dan una explicación directa de qué
características pesan más para distinguir comportamiento sospechoso — útil
tanto como punto de comparación honesto contra el LSTM/MLP como para
justificar decisiones de feature engineering.
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate

COLUMNAS_NUMERICAS = [
    "Desplazamiento", "Velocidad", "Aceleracion", "Direccion", "Densidad",
    "Linealidad", "Circularidad", "Zigzag", "Frecuencia_Ciclo",
    "Amplitud_Ciclo", "Area_Trayectoria", "Tiempo_Permanencia",
]
COLUMNAS_BINARIAS = ["Es_Ciclico", "En_Interaccion"]
COLUMNAS_CATEGORICAS = {
    "Patron_Movimiento": ["Lineal", "Circular", "Zigzag", "Mixto", "Indeterminado"],
    "Postura": ["Horizontal", "Vertical", "Neutral", "Desconocida"],
}


def resumir_video(df):
    """Convierte el CSV de un video (una fila por frame/objeto) en un único
    vector de características que resume todo el video."""
    resumen = {"n_filas": len(df), "n_objetos": df["Objeto"].nunique() if "Objeto" in df else 0}

    for col in COLUMNAS_NUMERICAS:
        if col not in df.columns:
            continue
        valores = pd.to_numeric(df[col], errors="coerce").fillna(0)
        resumen[f"{col}_media"] = valores.mean()
        resumen[f"{col}_std"] = valores.std() if len(valores) > 1 else 0.0
        resumen[f"{col}_max"] = valores.max()

    for col in COLUMNAS_BINARIAS:
        if col in df.columns:
            resumen[f"{col}_fraccion"] = pd.to_numeric(df[col], errors="coerce").fillna(0).mean()

    for col, categorias in COLUMNAS_CATEGORICAS.items():
        if col not in df.columns:
            continue
        total = len(df)
        for cat in categorias:
            resumen[f"{col}_{cat}_fraccion"] = (df[col] == cat).sum() / total if total > 0 else 0.0

    return resumen


def cargar_dataset(csv_dir, class_names=None):
    if class_names is None:
        class_names = ["normal", "merodeo", "forcejeo"]

    filas = []
    etiquetas = []
    nombres_video = []

    for filename in sorted(os.listdir(csv_dir)):
        if not filename.endswith(".csv"):
            continue

        prefijo = filename.split("_", 1)[0]
        if prefijo not in class_names:
            continue

        df = pd.read_csv(os.path.join(csv_dir, filename))
        if df.empty:
            continue

        filas.append(resumir_video(df))
        etiquetas.append(class_names.index(prefijo))
        nombres_video.append(filename)

    X = pd.DataFrame(filas).fillna(0)
    y = np.array(etiquetas)
    return X, y, nombres_video, class_names


def entrenar_y_evaluar(csv_dir, class_names=None, random_state=42, n_splits=5):
    """
    Evalúa cada modelo con validación cruzada estratificada sobre TODO el
    dataset, en vez de un único split 80/20. Con ~78 videos, un solo holdout
    de ~16 muestras da un accuracy con demasiada varianza para ser un número
    confiable (cada video vale ~6 puntos porcentuales); promediar sobre
    varios folds -y usar las predicciones out-of-fold de todos para la
    matriz de confusión- da una estimación mucho más honesta. El modelo que
    se guarda al final se reentrena con el 100% de los datos, ya que la
    validación cruzada es la que certifica qué tan bien generaliza.
    """
    X, y, nombres_video, class_names = cargar_dataset(csv_dir, class_names)
    print(f"Videos cargados: {len(X)}  |  Características por video: {X.shape[1]}")
    for i, nombre in enumerate(class_names):
        print(f"  {nombre}: {(y == i).sum()} videos")

    conteos = [int((y == i).sum()) for i in range(len(class_names))]
    n_splits_efectivo = min(n_splits, min(conteos)) if conteos else 0
    etiquetas_posibles = list(range(len(class_names)))

    modelos = {
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=2,
            class_weight="balanced", random_state=random_state
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=random_state
        ),
    }

    resultados = {}

    if n_splits_efectivo < 2:
        print(f"\nNo hay suficientes muestras por clase ({conteos}) para "
              f"validación cruzada; se necesitan al menos 2 por clase.")
        return resultados

    cv = StratifiedKFold(n_splits=n_splits_efectivo, shuffle=True, random_state=random_state)

    for nombre_modelo, modelo in modelos.items():
        scores = cross_validate(
            modelo, X, y, cv=cv, scoring=["accuracy", "f1_weighted"]
        )
        acc_mean, acc_std = scores["test_accuracy"].mean(), scores["test_accuracy"].std()
        f1_mean, f1_std = scores["test_f1_weighted"].mean(), scores["test_f1_weighted"].std()

        # Predicciones "out-of-fold": cada video se predice con un modelo
        # que nunca lo vio en entrenamiento, así que se puede armar un
        # classification_report/matriz de confusión sobre los 78 videos
        # completos sin fuga de datos.
        y_pred_oof = cross_val_predict(modelo, X, y, cv=cv)

        print(f"\n=== {nombre_modelo} (validación cruzada, {n_splits_efectivo}-fold) ===")
        print(f"Accuracy: {acc_mean:.3f} ± {acc_std:.3f}  |  F1 (weighted): {f1_mean:.3f} ± {f1_std:.3f}")
        print(f"Accuracy por fold: {[round(s, 3) for s in scores['test_accuracy']]}")
        print(classification_report(
            y, y_pred_oof, labels=etiquetas_posibles, target_names=class_names, zero_division=0
        ))
        print("Matriz de confusión out-of-fold (filas=real, columnas=predicho):")
        print(confusion_matrix(y, y_pred_oof, labels=etiquetas_posibles))

        # Modelo final: reentrenado con el 100% de los datos para guardarlo
        modelo.fit(X, y)
        if hasattr(modelo, "feature_importances_"):
            importancias = pd.Series(modelo.feature_importances_, index=X.columns)
            print("\nTop 10 características más importantes:")
            print(importancias.sort_values(ascending=False).head(10).to_string())

        modelo_path = f"best_model_{nombre_modelo}.pkl"
        joblib.dump({"modelo": modelo, "class_names": class_names, "columnas": list(X.columns)}, modelo_path)
        print(f"Modelo guardado en {modelo_path} (entrenado con el 100% de los datos)")

        resultados[nombre_modelo] = {
            "accuracy_mean": acc_mean, "accuracy_std": acc_std,
            "f1_mean": f1_mean, "f1_std": f1_std,
        }

    return resultados


def predecir_video(modelo_path, csv_path):
    """Predice la clase de un video a partir de su CSV de características,
    usando un modelo entrenado con esta misma función."""
    paquete = joblib.load(modelo_path)
    modelo = paquete["modelo"]
    class_names = paquete["class_names"]
    columnas = paquete["columnas"]

    df = pd.read_csv(csv_path)
    resumen = resumir_video(df)
    X = pd.DataFrame([resumen]).reindex(columns=columnas, fill_value=0)

    pred = modelo.predict(X)[0]
    proba = modelo.predict_proba(X)[0]

    return {
        "predicted_class": int(pred),
        "predicted_label": class_names[pred],
        "probabilities": {name: float(p) * 100 for name, p in zip(class_names, proba)},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenar/evaluar modelos de árboles sobre estadísticas por video")
    parser.add_argument("--csv_dir", type=str,
                         default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                               "..", "informacion", "csv"))
    args = parser.parse_args()

    entrenar_y_evaluar(args.csv_dir)
