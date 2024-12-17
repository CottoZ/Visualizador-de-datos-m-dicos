import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Crear el DataFrame manualmente con los datos de ejemplo
data = {
    'age': [22, 25, 28, 30, 35],
    'height': [165, 170, 175, 180, 185],
    'weight': [60, 65, 70, 75, 80],
    'ap_hi': [120, 130, 110, 140, 135],
    'ap_lo': [80, 85, 70, 90, 88],
    'cholesterol': [1, 2, 1, 2, 1],
    'gluc': [1, 1, 2, 1, 1],
    'smoke': [0, 1, 0, 0, 1],
    'alco': [0, 0, 1, 0, 1],
    'active': [1, 1, 1, 1, 1],
    'cardio': [0, 1, 0, 1, 0],
    'overweight': [0, 0, 0, 1, 0]
}

# Convertir el diccionario en un DataFrame de pandas
df = pd.DataFrame(data)

# Mostrar una descripción estadística básica del DataFrame
print("Descripción estadística básica del DataFrame:")
print(df.describe())

# Generar el mapa de calor de correlación
print("Generando el mapa de calor de correlación...")

correlation_matrix = df.corr()  # Calcular la correlación de los datos
correlation_matrix = correlation_matrix.round(2)  # Redondear a dos decimales

# Crear el mapa de calor utilizando seaborn
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', cbar=True)
plt.title('Correlación de los datos')
plt.show()

# Generar el gráfico de barras para la columna 'cholesterol'
print("Generando gráfico de barras de 'cholesterol'...")

sns.countplot(x='cholesterol', data=df)
plt.title("Distribución de Cholesterol")
plt.show()
