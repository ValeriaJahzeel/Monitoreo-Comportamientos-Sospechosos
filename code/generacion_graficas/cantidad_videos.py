import matplotlib.pyplot as plt

# Datos
clases = ['Normales', 'Forcejeo', 'Merodeo']
valores = [30, 22, 26]
colores = ['green', 'orange', 'blue']

# 1. Gráfica de barras
plt.figure(figsize=(8, 5))
bars = plt.bar(clases, valores, color=colores, linestyle='dashed')

# Add value labels on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}',
             ha='center', va='bottom')

plt.title('Distribución de videos por clase')
plt.ylabel('Cantidad de videos')
plt.tight_layout()
plt.show()

# 2. Gráfica de pastel (opcional)
plt.figure(figsize=(7, 7))
plt.pie(valores, labels=clases, autopct='%1.1f%%', colors=colores, startangle=90)
plt.title('Proporción de videos por clase')
plt.axis('equal')  # Círculo
plt.tight_layout()
plt.show()
