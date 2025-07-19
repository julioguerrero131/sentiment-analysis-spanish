import pandas as pd
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException

# Cargar el archivo CSV
df = pd.read_csv('2020-03-29 Coronavirus Tweets.CSV', nrows=2000)
print(df)

# Función para detectar el idioma de forma segura
def detectar_idioma(texto):
    try:
        return detect(str(texto))
    except LangDetectException:
        return "unknown"

# Aplicar la detección de idioma a la columna
df['idioma'] = df['text'].apply(detectar_idioma)

# Filtrar los tweets en español
tweets_es = df[df['idioma'] == 'es']

# Guardar resultados si lo necesitas
tweets_es.to_csv('tweets_espanol.csv', index=False)
