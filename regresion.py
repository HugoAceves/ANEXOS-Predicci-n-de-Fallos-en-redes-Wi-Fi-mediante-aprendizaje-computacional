import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Cargar dataset
df = pd.read_csv('/home/hugo/INFOTEC/TesisMCDI/Data/dataSanosUnicos.csv')


# Eliminar columna no relevante para el modelo
X = df.drop(['device_id', 'classification'], axis=1)


# Convertir la variable objetivo a binaria: 0 = "No event", 1 = otros eventos
y = np.where(df['classification'] == '(Event) No event', 0, 1)


# Dividir en datos de entrenamiento y prueba (70% - 30%)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    stratify=y,  # Mantener proporción de clases
    random_state=42,
    shuffle=True #Mezclamos los datos antes de dividirlos
)


# Verificación rápida
print("Distribución de clases en entrenamiento:", np.unique(y_train, return_counts=True))
print("Distribución de clases en prueba:", np.unique(y_test, return_counts=True))
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Entrenar modelo con class_weight balanceado
lr = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
lr.fit(X_train, y_train)


# Predecir y evaluar
y_pred = lr.predict(X_test)


print("=== Regresión Logística ===")
print(f"Exactitud: {accuracy_score(y_test, y_pred):.2f}")
print("\nMatriz de confusión:")
print(confusion_matrix(y_test, y_pred))
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))
