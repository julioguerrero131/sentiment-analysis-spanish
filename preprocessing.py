import pandas as pd

#### CANTIDAD DE POSITIVOS, NEGATIVOS Y NEUTRALES

# Cargar el CSV
df = pd.read_csv('data/ia_tweets_500.csv')

# Contar la cantidad de tweets por polaridad
conteo_polaridades = df['polarity'].value_counts()

# Obtener los valores individuales
P = conteo_polaridades.get('P', 0)
N = conteo_polaridades.get('N', 0)
NEU = conteo_polaridades.get('NEU', 0)

# Mostrar resultados
print(f"Positivos: {P}")
print(f"Negativos: {N}")
print(f"Neutrales: {NEU}")



#### OBTENER NUEVO ARCHIVO CSV

# # Cargar el CSV original
# df = pd.read_csv('data/ia_tweets.csv')

# # Filtrar los primeros 500 de cada tipo de polaridad
# positivos = df[df['polarity'] == 'P'].head(500)
# negativos = df[df['polarity'] == 'N'].head(500)
# neutrales = df[df['polarity'] == 'NEU'].head(500)

# # Concatenar los tres DataFrames
# resultado = pd.concat([positivos, negativos, neutrales])

# # Guardar en un nuevo archivo CSV
# resultado.to_csv('data/ia_tweets_500.csv', index=False)

# print("Archivo generado: ia_tweets_500.csv")
