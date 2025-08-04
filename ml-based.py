import pandas as pd
from pysentimiento import create_analyzer
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# Activar barra de progreso en pandas
tqdm.pandas()

# Crear el analizador de sentimientos en español
analyzer = create_analyzer(task="sentiment", lang="es")

# === Leer el CSV ===
df = pd.read_csv("data/ia_tweets_500.csv")

# Mapeo de etiquetas originales
mapeo_polaridad = {
    "P": "POS",
    "N": "NEG",
    "NEU": "NEU"
}

# Filtrar y mapear polaridades válidas
df = df[df["polarity"].isin(mapeo_polaridad.keys())]
df["label"] = df["polarity"].map(mapeo_polaridad)

# === Clasificación con pysentimiento ===
def clasificar_sentimiento(texto):
    try:
        return analyzer.predict(texto).output
    except Exception as e:
        return "ERROR"

print("Clasificando tweets con pysentimiento...")
df["pred"] = df["text"].progress_apply(clasificar_sentimiento)

# Eliminar errores
df = df[df["pred"] != "ERROR"]

# === Evaluación ===
etiquetas_modelo = ["POS", "NEU", "NEG"]
etiquetas_visuales = ["Positivo", "Neutro", "Negativo"]

# === Métricas ===
print("\n=== MÉTRICAS DE CLASIFICACIÓN ===")

# Matriz de Confusión
cm = confusion_matrix(df["label"], df["pred"], labels=etiquetas_modelo)

# Reporte completo
reporte = classification_report(df["label"], df["pred"], labels=etiquetas_modelo, zero_division=0)
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
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=etiquetas_visuales, yticklabels=etiquetas_visuales)
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title("Matriz de Confusión - pysentimiento")
plt.tight_layout()
plt.show()
