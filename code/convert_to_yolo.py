def convert_to_yolo(input_file, output_file, image_width, image_height):
    # Abre el archivo de entrada y el de salida
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        lines = infile.readlines()

        for line in lines:
            # Elimina espacios en blanco y saltos de línea
            line = line.strip()

            # Si la línea está vacía, continúa con la siguiente línea
            if not line:
                continue

            # Divide la línea por comas
            data = line.split(',')

            # Verifica si la línea tiene la cantidad correcta de datos (12 valores)
            if len(data) != 12:
                print(f"Advertencia: Línea ignorada, no tiene el formato correcto: {line}")
                continue  # Si no tiene 12 valores, la ignoramos

            # Extrae los valores del texto (como flotantes)
            try:
                head_left = float(data[4])
                head_top = float(data[5])
                head_right = float(data[6])
                head_bottom = float(data[7])

                body_left = float(data[8])
                body_top = float(data[9])
                body_right = float(data[10])
                body_bottom = float(data[11])
            except ValueError:
                print(f"Advertencia: Línea ignorada, contiene valores no numéricos: {line}")
                continue  # Si hay un error en la conversión a flotante, ignoramos la línea

            # Calcula los centros y tamaños normalizados para la cabeza
            head_x_center = (head_left + head_right) / 2 / image_width
            head_y_center = (head_top + head_bottom) / 2 / image_height
            head_width = (head_right - head_left) / image_width
            head_height = (head_bottom - head_top) / image_height

            # Calcula los centros y tamaños normalizados para el cuerpo
            body_x_center = (body_left + body_right) / 2 / image_width
            body_y_center = (body_top + body_bottom) / 2 / image_height
            body_width = (body_right - body_left) / image_width
            body_height = (body_bottom - body_top) / image_height

            # Escribe las filas correspondientes al formato YOLO (class_id, x_center, y_center, width, height)
            # Asignamos 0 para la cabeza y 1 para el cuerpo
            outfile.write(f"0 {head_x_center} {head_y_center} {head_width} {head_height}\n")
            outfile.write(f"1 {body_x_center} {body_y_center} {body_width} {body_height}\n")
            
            print(f"Datos convertidos: {line}")

# Llama a la función proporcionando los parámetros adecuados
input_file = 'data.txt'  # Archivo de entrada en formato .txt
output_file = 'output_yolo.txt'  # Archivo de salida en formato YOLO
image_width = 640  # Ancho de la imagen
image_height = 480  # Alto de la imagen

convert_to_yolo(input_file, output_file, image_width, image_height)
