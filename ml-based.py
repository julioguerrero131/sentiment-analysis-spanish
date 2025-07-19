import pandas as pd
from pysentimiento import create_analyzer
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Crear el analizador de sentimientos en español
analyzer = create_analyzer(task="sentiment", lang="es")

# --------------------
# Dataset de ejemplo
# --------------------
# Lista de tweets con su sentimiento real (etiqueta)
data = [
    {"tweet": "Me encanta este producto, es genial", "label": "POS"},
    {"tweet": "Este servicio fue terrible", "label": "NEG"},
    {"tweet": "No estuvo mal, pero tampoco excelente", "label": "NEU"},
    {"tweet": "Odio cuando pasa esto", "label": "NEG"},
    {"tweet": "Estoy feliz con los resultados", "label": "POS"},
    {"tweet": "Todo fue como esperaba", "label": "NEU"},
    {"tweet": "La atención al cliente fue pésima", "label": "NEG"},
    {"tweet": "Qué maravilla de experiencia", "label": "POS"},
    {"tweet": "Nada especial, fue común", "label": "NEU"},
    {"tweet": "Nunca volveré a usar esto", "label": "NEG"}
]

df = pd.DataFrame(data)

# --------------------------
# Obtener predicciones
# --------------------------
# Predecimos el sentimiento para cada tweet
df["pred"] = df["tweet"].apply(lambda x: analyzer.predict(x).output)

# --------------------------
# Evaluación
# --------------------------
# Convertimos a listas
y_true = df["label"].tolist()
y_pred = df["pred"].tolist()

# Generar matriz de confusión
cm = confusion_matrix(y_true, y_pred, labels=["POS", "NEU", "NEG"])
labels = ["Positivo", "Neutro", "Negativo"]

# Mostrar matriz de confusión con seaborn
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title("Matriz de Confusión - Análisis de Sentimiento")
plt.tight_layout()
plt.show()

# Reporte detallado
print("Reporte de clasificación:\n")
print(classification_report(y_true, y_pred, target_names=labels))
