# Algoritmo de detección de patrones de movimiento para la identificación de allanamiento de morada usando visión computacional

El objetivo de este proyecto es desarrollar un algoritmo de detección de comportamientos sospechosos en videos de vigilancia de casa habitación, enfocándose en el merodeo y los intentos de allanamiento. 
Se recopilaron videos del UCF Crime Dataset y YouTube, mejorando su calidad mediante ecualización de histograma. Utilizando el flujo óptico, se analizaron movimientos en los cuadros para identificar posiciones y trayectorias clave. El algoritmo utiliza redes neuronales convolucionales (CNN) para la identificar personas y manos, también utiliza un perceptron multi- calpa (MLP) para clasificar comportamientos en merodeo (permanencia mayor a 15 segundos) y forzar accesos (movimientos sospechosos cerca de puertas o ventanas).

## Comenzando 🚀

Solo clona el proyecto en tu repositorio y descarga el dataset del siguiente link de drive:

[aqui segun va el link xd]

### Pre-requisitos 📋

Debes tener instaladas las siguientes bibiliotecas
* cv2
```
pip install opencv-python
```
* numpy
```
pip install numpy
```

## Construido con 🛠️

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

## Autores ✒️

* **Valeria Castañón** - *Trabajo Inicial* - [ValeriaJahzeel](https://github.com/ValeriaJahzeel)

