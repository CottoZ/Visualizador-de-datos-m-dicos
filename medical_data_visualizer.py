import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Cargar los datos
df = pd.read_csv('medical_examination.csv')

# Agregar columna 'overweight'
df['overweight'] = (df['weight'] / (df['height'] / 100) ** 2).apply(lambda x: 1 if x > 25 else 0)

# Asegurarnos de que 'id' y 'sex' están presentes en el DataFrame
df['id'] = df.index  # Agregar la columna 'id' si es necesario

# Función para generar el gráfico de barras
def draw_cat_plot():
    # Convertir el dataframe a formato largo
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # Agrupar los datos por 'cardio', 'variable' y 'value', y contar el número de elementos
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')

    # Eliminar cualquier combinación que no tenga un valor significativo
    df_cat = df_cat[df_cat['total'] > 0]

    # Renombrar la columna 'value' a 'level' para hacerla más entendible en el gráfico
    df_cat = df_cat.rename(columns={'value': 'level'})

    # Crear el gráfico usando seaborn
    g = sns.catplot(x='variable', hue='level', col='cardio', data=df_cat, kind='count')

    # Cambiar la etiqueta del eje x a 'variable'
    g.set_axis_labels('variable', 'total')

    # Guardar la figura como un archivo PNG
    g.savefig('catplot.png')
    
    return g.fig  # Retornar la figura generada

# Función para generar el mapa de calor
def draw_heat_map():
    # Crear el DataFrame 'df_heat' solo con las columnas necesarias para el análisis
    df_heat = df[['id', 'age', 'sex', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio', 'overweight']]

    # Eliminar las filas con valores nulos
    df_heat = df_heat.dropna()

    # Verificar que no haya filas o columnas vacías
    print("Datos antes de la correlación:")
    print(df_heat.isnull().sum())  # Ver si hay columnas con valores nulos

    # Calcular la matriz de correlación
    corr = df_heat.corr()

    # Redondear los valores de la correlación a 1 decimal
    corr = corr.round(1)

    # Convertir los valores negativos cero a cero, para evitar -0.0
    corr = corr.applymap(lambda x: 0.0 if np.isclose(x, 0.0) else x)

    # Eliminar cualquier fila o columna con solo valores cero (si los hay)
    corr = corr.loc[(corr != 0).any(axis=1), (corr != 0).any(axis=0)]

    # Crear una máscara para ocultar la mitad superior de la matriz de correlación
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Crear el gráfico de calor con seaborn
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".1f", cmap='coolwarm', cbar_kws={'shrink': 0.8}, ax=ax)

    # Guardar el gráfico de calor como un archivo PNG
    fig.savefig('heatmap.png')

    return fig  # Retornar la figura generada
