# Importar librerías
from ultralytics import YOLO  # Importar el modelo YOLO de Ultralytics
import cv2                   # Importar la biblioteca OpenCV
import math                  # Importar el módulo math para operaciones matemáticas


""" 
Notas:
    * Trabajar para ver como juntar lo modelos de mano y persona 
    * Darle mas entrenamiento al modelo ya que no detecta bien a personas volteadas o desde el angulo arriba
    * A veces confunde mano y persona
    
    * El entrenamiento de manos se realizó con 10 epocas
"""
# Ruta del archivo de video
video_path = "./dataset/sospechoso/1.mp4"  # Cambia esta ruta por la ubicación de tu archivo de video

def modelo(video_path):
    cap = cv2.VideoCapture(video_path)   
    
    if not cap.isOpened():  # Verificar si el video se abre correctamente
        print("Error: No se puede abrir el video.")
        exit()

    # Cargar el modelo YOLO
    model = YOLO("yolo-Weights/yolov8n.pt")  # Cargar el modelo YOLOv8 con pesos pre-entrenados

    model = YOLO("./runs/detect/train4/weights/best.pt")

    # Definir las clases de objetos para la detección
    classNames = ["persona", "bicicleta", "coche", "motocicleta", "avión", "autobús", "tren", "camión", "bote",
                "semáforo", "boca de incendios", "señal de stop", "parquímetro", "banco", "pájaro", "gato",
                "perro", "caballo", "oveja", "vaca", "elefante", "oso", "cebra", "jirafa", "mochila", "paraguas",
                "bolso", "corbata", "maleta", "frisbee", "esquís", "snowboard", "balón deportivo", "cometa", "bate de béisbol",
                "guante de béisbol", "monopatín", "tabla de surf", "raqueta de tenis", "botella", "copa de vino", "taza",
                "tenedor", "cuchillo", "cuchara", "cuenco", "plátano", "manzana", "sándwich", "naranja", "brócoli",
                "zanahoria", "perrito caliente", "pizza", "donut", "pastel", "silla", "sofá", "planta en maceta", "cama",
                "mesa de comedor", "retrete", "monitor de TV", "ordenador portátil", "ratón", "control remoto", "teclado", "teléfono móvil",
                "microondas", "horno", "tostadora", "fregadero", "refrigerador", "libro", "reloj", "florero", "tijeras",
                "oso de peluche", "secador de pelo", "cepillo de dientes", "mano"
                ]

    claseDeseada = ["persona",  "mano"]


    # Bucle para capturar fotogramas del video
    while True:
        # Leer un fotograma del video
        success, img = cap.read()

        # Si no se puede leer el fotograma, el video ha terminado
        if not success:
            break

        # Realizar la detección de objetos utilizando el modelo YOLO en el fotograma capturado
        results = model(img, stream=True)

        # Iterar a través de los resultados de la detección de objetos
        for r in results:
            boxes = r.boxes  # Extraer las cajas delimitadoras de los objetos detectados

            # Iterar a través de cada caja delimitadora
            for box in boxes:
                # Extraer coordenadas de la caja delimitadora
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convertir a valores enteros

                # Dibujar la caja delimitadora en el fotograma
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # Calcular e imprimir la puntuación de confianza de la detección
                confidence = math.ceil((box.conf[0]*100))/100

                # Determinar e imprimir el nombre de la clase del objeto detectado
                cls = int(box.cls[0])
                
                if classNames[cls] in claseDeseada:
                    print(f"Detectado: {classNames[cls]} con {confidence} de confianza")

                    # Dibujar la caja en la imagen
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                    cv2.putText(img, classNames[cls], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (255, 0, 0), 2)

                    # Dibujar texto indicando el nombre de la clase en el fotograma
                    org = [x1, y1]
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    fontScale = 1
                    color = (255, 0, 0)
                    thickness = 2
                    cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

        # Mostrar el fotograma con los objetos detectados en una ventana llamada "Video"
        cv2.imshow('Video', img)

        # Comprobar si se presionó la tecla 'q' para salir del bucle
        if cv2.waitKey(1) == ord('q'):
            break

    # Liberar el video
    cap.release()

    # Cerrar todas las ventanas de OpenCV
    cv2.destroyAllWindows()
