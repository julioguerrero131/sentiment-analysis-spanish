import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from deep_translator import GoogleTranslator
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

#nltk.download('vader_lexicon')

# Crear el analizador
analyzer = SentimentIntensityAnalyzer()

# Función para clasificar con VADER traducido
def vader_es(texto_es):
    try:
        texto_en = GoogleTranslator(source='es', target='en').translate(texto_es)
        score = analyzer.polarity_scores(texto_en)['compound']
        if score >= 0.05:
            return "positivo"
        elif score <= -0.05:
            return "negativo"
        else:
            return "neutral"
    except Exception as e:
        return "error"

# === Datos de prueba ===
textos = [
    "Estoy muy feliz con el resultado",
    "Esto es terrible y me siento mal",
    "Está bien, nada fuera de lo normal",
    "Me encanta este lugar",
    "Odio cuando pasa esto",
    "La experiencia fue regular"
]

etiquetas_reales = ["positivo", "negativo", "neutral", "positivo", "negativo", "neutral"]

# === Clasificación ===
predicciones = [vader_es(texto) for texto in textos]

# === Matriz de Confusión ===
etiquetas = ["negativo", "neutral", "positivo"]
matriz = confusion_matrix(etiquetas_reales, predicciones, labels=etiquetas)

# Visualización
plt.figure(figsize=(6, 4))
sns.heatmap(matriz, annot=True, fmt='d', cmap='Blues', xticklabels=etiquetas, yticklabels=etiquetas)
plt.title("Matriz de Confusión - VADER con traducción")
plt.xlabel("Predicho")
plt.ylabel("Real")
plt.tight_layout()
plt.show()
