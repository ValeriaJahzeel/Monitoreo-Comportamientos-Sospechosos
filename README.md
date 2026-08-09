# 🕵️ Suspicious Behavior Detection System

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.13-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![YOLO](https://img.shields.io/badge/YOLOv8-purple.svg)](https://ultralytics.com/)

> Computer vision pipeline that watches short surveillance clips and classifies what's happening as **normal activity**, **loitering (merodeo)**, or **forced entry (forcejeo)** — built end-to-end from raw video to a trained, cross-validated classifier.

---

## 📋 Overview

| Behavior | Description | Risk level |
|---|---|---|
| 🟢 **Normal** | Regular activity — deliveries, routine entries | Low |
| 🟡 **Merodeo** (loitering) | Repetitive, lingering movement near the property | Medium |
| 🔴 **Forcejeo** (forced entry) | Abrupt, forceful action targeting doors/windows | High |

The pipeline: **YOLOv8** (person detection) → a **from-scratch SORT-style tracker** (Kalman filter + Hungarian/IoU matching) → **19 hand-engineered kinematic/behavioral features per tracked person per frame** → a classifier (compared across 4 model families, evaluated with 5-fold cross-validation).

Dataset: **78 real surveillance clips** (30 normal, 26 merodeo, 22 forcejeo), ranging from a few seconds to several minutes each.

---

## 📊 Results (5-fold stratified cross-validation, 78 videos)

A single 80/20 train/test split on a dataset this size leaves the validation set at ~16 videos — each one is worth several percentage points of "accuracy," which makes a single-split number close to meaningless. Every result below is the mean ± standard deviation across 5 stratified folds, using **all 78 videos** (never mixed between train and validation within a fold).

| Model | Accuracy | F1 (weighted) | Notes |
|---|---|---|---|
| 🏆 **Gradient Boosting** | **71.8% ± 9.5%** | 0.714 ± 0.105 | Best mean accuracy |
| **Random Forest** | 70.3% ± 11.7% | 0.705 ± 0.116 | Close second |
| **LSTM** (bidirectional + attention) | 60.2% ± 10.8% | 0.577 ± 0.109 | Struggles most on *forcejeo* (27% recall) |
| **MLP** (mean+max pooling) | 57.7% ± 3.2% | 0.556 ± 0.042 | Lowest variance across folds |

*(random-chance baseline with 3 balanced classes: ~33%)*

**Honest takeaway:** with ~78 videos, the two tree ensembles (trained on per-video aggregated statistics) generalize noticeably better than the sequence models (trained on the full per-frame sequence). This is a real, cross-validated result, not an artifact of a lucky split — it's also a good demonstration that "bigger model" isn't automatically "better model" when the dataset is small.

The most important engineered feature across both tree models is `Linealidad` (trajectory straightness); `Tiempo_Permanencia` (dwell time — see below) consistently ranks in the top few features, confirming that "how long someone lingers in place" is genuine signal for loitering detection.

---

## 🏗️ Pipeline architecture

```
 video ─▶ YOLOv8 (person-only, class=0)
            │
            ▼
   SORT-style tracker (code/tracking.py)
   Kalman filter (constant-velocity) + IoU/Hungarian association
   tolerates brief occlusion instead of spawning a new ID
            │
            ▼
   Dense optical flow (Farneback) — fallback only,
   used solely on frames where YOLO found no one
            │
            ▼
   Feature extraction (code/featureExtraction.py)
   19 features / frame / tracked person
            │
            ▼
   ┌─────────────────────┬─────────────────────┐
   │ LSTM / MLP           │ Random Forest /      │
   │ (code/model.py)      │ Gradient Boosting    │
   │ full per-frame        │ (code/entrenamiento/ │
   │ sequence               │ modelo_arboles.py)   │
   └─────────────────────┴─────────────────────┘
            │
            ▼
   normal / merodeo / forcejeo + confidence
```

### Extracted features (per frame, per tracked person)

`Centroide_X/Y`, `Desplazamiento`, `Velocidad`, `Aceleracion`, `Direccion`, `Densidad` (motion density), `Postura` (bbox aspect ratio), `Patron_Movimiento` (linear/circular/zigzag/mixed), `Linealidad`, `Circularidad`, `Zigzag`, `Es_Ciclico` + `Frecuencia_Ciclo` + `Amplitud_Ciclo` (periodicity via autocorrelation), `Area_Trayectoria` (convex hull of the trajectory), `En_Interaccion` (proximity to another tracked person), and **`Tiempo_Permanencia`** — consecutive frames a person has stayed nearly stationary, used as a proxy for "dwell time" near a property without needing to hand-annotate where a door or window is in each scene.

---

## 🛠️ Installation

```bash
git clone https://github.com/ValeriaJahzeel/Monitoreo-Comportamientos-Sospechosos.git
cd Monitoreo-Comportamientos-Sospechosos

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

**GPU note:** PyTorch's CUDA wheels currently lag behind the newest Python releases. If you're on **Python 3.13**, `pip install torch` gives you a CPU-only build. For CUDA acceleration, use **Python 3.11 or 3.12** and install torch from the CUDA index, e.g.:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
On this project's dev machine (RTX 3050, 4GB), that made feature extraction ~5x faster than CPU — most of that gain actually came from skipping unnecessary work (dense optical flow and frame annotation were being computed on every frame even when not needed), not from the GPU alone.

---

## 🚀 Quick start

### Classify a video (the demo script)

```bash
python code/predecir.py --video path/to/video.mp4
python code/predecir.py --video path/to/video.mp4 --modelo random_forest
python code/predecir.py --video path/to/video.mp4 --guardar_video anotado.mp4
```

This runs the full pipeline (detection → tracking → feature extraction → classification) on a single video and prints the predicted class with per-class probabilities. `--guardar_video` additionally saves an annotated copy (bounding boxes, IDs, trajectories) for visual demos.

### Extract features from a dataset

```python
from code.objectDetection import ObjectDetector

detector = ObjectDetector()
detector.procesar_video("dataset/merodeo/3.mp4", "informacion/csv/merodeo_3.csv")
```

CSV filenames must be prefixed with the class name (`normal_`, `merodeo_`, `forcejeo_`) — that prefix is how the training code assigns labels.

### Train a model

```bash
# LSTM or MLP, single train/val split
python code/model.py --mode train --model_type lstm --csv_dir informacion/csv
python code/model.py --mode train --model_type mlp  --csv_dir informacion/csv

# LSTM or MLP, proper 5-fold cross-validation (recommended for reporting results)
python code/model.py --mode cross_validate --model_type lstm --csv_dir informacion/csv --n_splits 5

# Tree models (always cross-validated, trains in seconds)
python code/entrenamiento/modelo_arboles.py --csv_dir informacion/csv
```

### Predict with a trained model directly

```python
import code.model as model

modelo, checkpoint = model.load_best_model("best_model_lstm.pth")
scaler = model.load_scaler_from_checkpoint(checkpoint)
resultado = model.predict_video(modelo, "informacion/csv/nuevo_video.csv", scaler=scaler,
                                 class_names=checkpoint["class_names"])
```

---

## 📁 Project structure

```
Monitoreo-Comportamientos-Sospechosos/
├── code/
│   ├── objectDetection.py      # YOLOv8 detection + SORT tracking + orchestration
│   ├── tracking.py             # Kalman filter + IoU/Hungarian tracker, built from scratch
│   ├── featureExtraction.py    # Per-frame feature computation
│   ├── model.py                # LSTM / MLP: dataset, training, cross-validation, inference
│   ├── predecir.py             # End-to-end demo: video in, prediction out
│   ├── analysis.py             # Exploratory analysis (clustering, PCA, outlier detection)
│   ├── filtrarCaracteristicas.py  # Optional feature-set reduction
│   ├── generador_forcejeo.py   # ⚠️ Synthetic data generator — see warning in the file itself
│   └── entrenamiento/
│       └── modelo_arboles.py   # Random Forest / Gradient Boosting + cross-validation
├── dataset/                    # normal/, merodeo/, forcejeo/ — source videos (not versioned)
├── informacion/csv/            # Extracted features, one CSV per video (not versioned)
├── requirements.txt
└── README.md
```

---

## ⚠️ Known limitations

- **Small dataset.** 78 videos is enough to get a real, cross-validated signal above chance, but not enough to claim production-grade accuracy. Every number above has a wide confidence interval — treat the ranking (trees > sequence models) as more reliable than the exact percentages.
- **LSTM confuses forcejeo with merodeo.** Out-of-fold recall for *forcejeo* is only 27% for the LSTM (vs. ~65-77% for the other two classes) — both behaviors can look similar in raw trajectory shape over short clips.
- **Tracking still isn't perfect.** The custom SORT tracker massively reduced spurious ID churn (see commit history / PR description for before/after numbers), but fast motion or heavy occlusion in busier clips can still fragment a single person into multiple track IDs.
- **`generador_forcejeo.py` produces synthetic data**, not real footage — it's kept in the repo but explicitly flagged; don't mix its output into reported metrics without saying so.

---

## 🗺️ Roadmap

The long-term goal of this project is a model light enough to run on **cheap, non-specialized home security cameras with no GPU**, usable by anyone rather than a security team with dedicated hardware — and flexible enough to be **retrained for other kinds of suspicious behavior**, not just loitering/forced entry. Where things stand against that goal, and what's next:

- **Lightweight inference is already mostly there.** The winning models (Random Forest / Gradient Boosting) are cheap at inference time — no GPU, milliseconds per prediction, a few MB on disk. The real bottleneck for genuinely low-power hardware is **YOLOv8n + the Python/PyTorch runtime** for the detection step. Next step: export the detector to an edge format (ONNX / TFLite, INT8 quantized) or evaluate an even smaller person-detector, and benchmark on hardware closer to what a real home camera uses.
- **Generalizing beyond merodeo/forcejeo.** The current system is a *supervised* classifier — adding a new behavior means recording and labeling new footage for it (the code already supports arbitrary class sets via `class_names`, so this is mostly a data problem, not a code problem). That doesn't scale to "any business, any situation" on its own. The more flexible path is reframing this as **anomaly detection**: learn what "normal" looks like per camera/installation and flag deviations, instead of learning fixed named categories. `analysis.py` already imports `IsolationForest` for exactly this but it was never wired into the main pipeline — that's the natural starting point.

---

## 📜 License

MIT License - see the [LICENSE](LICENSE) file for details.
