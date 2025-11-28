# ============================
# Parametre C i la seva influència en les mètriques
# ============================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# --- Cargar splits ---
train = pd.read_csv(r"C:\Users\clara\Documentos\3º GED\AC\proyecto copia\AnalizarLimpiarDividir\twitter_trainBALANCED.csv")
val = pd.read_csv(r"C:\Users\clara\Documentos\3º GED\AC\proyecto copia\AnalizarLimpiarDividir\twitter_valBALANCED.csv")
test = pd.read_csv(r"C:\Users\clara\Documentos\3º GED\AC\proyecto copia\AnalizarLimpiarDividir\twitter_testBALANCED.csv")

# --- Vectorización TF-IDF mejorada ---
vectorizer = TfidfVectorizer(max_features=80000, ngram_range=(1,3), sublinear_tf=True,min_df=2,max_df=0.95 )
X_train = vectorizer.fit_transform(train['text'])
X_val = vectorizer.transform(val['text'])
X_test = vectorizer.transform(test['text'])

y_train = train['label']
y_val = val['label']
y_test = test['label']

# Valores de C a probar
C_values = [0.001, 0.01, 0.1, 1, 2, 3, 5, 10]

train_scores = []
val_scores = []

# Entrenar y evaluar para cada valor de C
for C in C_values:
    logreg = LogisticRegression(
        C=C,
        solver='liblinear',
        class_weight='balanced',
        max_iter=300,
        random_state=42
    )
    logreg.fit(X_train, y_train)
    
    # Accuracy en train y validation
    train_scores.append(accuracy_score(y_train, logreg.predict(X_train)))
    val_scores.append(accuracy_score(y_val, logreg.predict(X_val)))

# --- Graficar curva de validación ---
plt.figure(figsize=(8,6))
plt.semilogx(C_values, train_scores, label="Train Accuracy", marker="o")
plt.semilogx(C_values, val_scores, label="Validation Accuracy", marker="o")

plt.xlabel("C (Regularization strength, log scale)")
plt.ylabel("Accuracy")
plt.title("Validation Curve - Logistic Regression")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("validation_curve_C.png", dpi=300)
plt.show()

print("Curva de validación guardada en 'validation_curve_C.png'")

