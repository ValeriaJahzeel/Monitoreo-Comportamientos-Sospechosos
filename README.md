# Algoritmo de Detección de Patrones de Movimiento para Identificación de Allanamiento de Morada

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 Descripción

Este proyecto desarrolla un algoritmo de detección de comportamientos sospechosos en videos de videovigilancia utilizando visión computacional y aprendizaje automático. El sistema es capaz de identificar tres tipos de comportamientos:

- **Normal**: Actividades cotidianas como llegada de repartidores o entrada normal al domicilio
- **Merodeo**: Movimientos repetitivos sin propósito aparente en las inmediaciones de un domicilio
- **Forcejeo**: Acciones bruscas centradas en puertas o ventanas que pueden indicar intentos de intrusión

## 🎯 Características Principales

- **Detección de personas** usando YOLO v3
- **Seguimiento de objetos** con algoritmo similar a SORT
- **Extracción de flujo óptico denso** para capturar movimientos sutiles
- **Agrupamiento con DBSCAN** para reducir falsos positivos
- **Múltiples arquitecturas** de redes neuronales (MLP, LSTM, Bi-LSTM)
- **Análisis de 36 características** cinemáticas y de trayectoria

## 🏗️ Arquitectura del Sistema

## 🚀 Resultados

| Modelo | Precisión | F1-Score Macro | Recall |
|--------|-----------|----------------|---------|
| **MLP** | **69%** | **0.54** | **66%** |
| LSTM Unidireccional | 43% | 0.43 | 41% |
| LSTM Bidireccional | 72% | 0.73 | 72% |

## 📊 Características Extraídas

El sistema analiza las siguientes características de movimiento:

### Características Cinemáticas
- Velocidad y aceleración
- Desplazamiento entre frames
- Dirección del movimiento
- Densidad de píxeles en movimiento

### Características de Trayectoria
- Linealidad y circularidad
- Patrones de zigzag
- Movimientos cíclicos
- Área cubierta por la trayectoria

### Características de Comportamiento
- Postura (horizontal/vertical)
- Interacción con objetos
- Tiempo de permanencia

## 🛠️ Instalación

### Prerrequisitos
```bash
Python 3.8+
CUDA 10.1+ (opcional, para GPU)

