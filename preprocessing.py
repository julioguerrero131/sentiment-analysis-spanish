import pandas as pd
import re

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



### OBTENER NUEVO ARCHIVO CSV

# Función para limpiar texto
def limpiar_texto(texto):
    # Eliminar URLs
    texto = re.sub(r'http\S+|www\.\S+', '', texto)
    # Eliminar menciones
    texto = re.sub(r'@\w+', '', texto)
    # Eliminar múltiples espacios
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

# Función para contar tokens significativos (al menos 3)
def es_valido(texto):
    tokens = re.findall(r'\b\w+\b', texto)
    return len(tokens) >= 3

# Cargar el CSV original
df = pd.read_csv('data/ia_tweets.csv')

# Limpiar texto
df['text'] = df['text'].astype(str).apply(limpiar_texto)

# Filtrar solo los que tengan 3 o más tokens significativos
df = df[df['text'].apply(es_valido)]

# Filtrar los primeros 500 de cada tipo de polaridad
positivos = df[df['polarity'] == 'P'].head(500)
negativos = df[df['polarity'] == 'N'].head(500)
neutrales = df[df['polarity'] == 'NEU'].head(500)

# Concatenar los tres DataFrames
resultado = pd.concat([positivos, negativos, neutrales])

# Guardar en un nuevo archivo CSV
resultado.to_csv('data/ia_tweets_500.csv', index=False)

print("Archivo generado: ia_tweets_500.csv")