from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# Barra de Carga
tqdm.pandas() 

# Crear el analizador
analyzer = SentimentIntensityAnalyzer()

# Función para clasificar con VADER traducido
def vader_es(texto_es):
    try:
        score = analyzer.polarity_scores(texto_es)['compound']
        if score >= 0.05:
            return "positivo"
        elif score <= -0.05:
            return "negativo"
        else:
            return "neutral"
    except Exception as e:
        return "error vader_es"

# === Leer el CSV ===
# Asegúrate de poner la ruta correcta al archivo
df = pd.read_csv("data/ia_tweets_500.csv")

# Solo nos quedamos con los campos necesarios
# Mapeamos la polaridad original a texto
mapeo_polaridad = {
    "P": "positivo",
    "N": "negativo",
    "NEU": "neutral"
}

# Filtramos y limpiamos
df = df[df["polarity"].isin(mapeo_polaridad.keys())]
df["label"] = df["polarity"].map(mapeo_polaridad)

# === Clasificar con VADER ===
print("Clasificando tweets con VADER en español...")
df["pred"] = df["text"].progress_apply(vader_es)

# Eliminamos filas donde ocurrió un error
df = df[df["pred"] != "error"]

# === Métricas ===
print("\n=== MÉTRICAS DE CLASIFICACIÓN ===")

# === Matriz de Confusión ===
etiquetas = ["negativo", "neutral", "positivo"]
print("Creando la matriz...")
matriz = confusion_matrix(df["label"], df["pred"], labels=etiquetas)

# Reporte completo
reporte = classification_report(df["label"], df["pred"], labels=etiquetas, zero_division=0)
print(reporte)

# Métricas individuales
accuracy = accuracy_score(df["label"], df["pred"])
precision = precision_score(df["label"], df["pred"], average='macro', zero_division=0)
recall = recall_score(df["label"], df["pred"], average='macro', zero_division=0)
f1 = f1_score(df["label"], df["pred"], average='macro', zero_division=0)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-Score : {f1:.4f}")

# Visualización
print("Matriz finalizada...")
plt.figure(figsize=(6, 4))
sns.heatmap(matriz, annot=True, fmt='d', cmap='Blues', xticklabels=etiquetas, yticklabels=etiquetas)
plt.title("Matriz de Confusión - VADER Español")
plt.xlabel("Predicho")
plt.ylabel("Real")
plt.tight_layout()
plt.show()