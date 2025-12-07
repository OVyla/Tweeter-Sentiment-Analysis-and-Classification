import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from AnalizarLimpiarDividir.vector_representation import load_and_vectorize_splits

# Load vectorized data (TF-IDF)
data = load_and_vectorize_splits(method='TFIDF')
X_train_full = data["X_train"]
X_val_full   = data["X_val"]
X_test_full  = data["X_test"]
y_train_full = data["y_train"]
y_val_full   = data["y_val"]
y_test_full  = data["y_test"]

# Sample 50%
# We use the index from y_train sample to slice X_train
train_sample = pd.Series(y_train_full).sample(frac=0.5, random_state=42)
val_sample = pd.Series(y_val_full).sample(frac=0.5, random_state=42)
test_sample = y_test_full.sample(frac=0.5, random_state=42)

X_train = X_train_full[train_sample.index]
y_train = train_sample.reset_index(drop=True)

X_val = X_val_full[val_sample.index]
y_val = val_sample.reset_index(drop=True)

X_test = X_test_full[test_sample.index]
y_test = test_sample.reset_index(drop=True)

# Train KNN model
knn = KNeighborsClassifier(
    n_neighbors=30,           # Número impar
    metric='cosine',         # ¡ESENCIAL! Cosine similarity para TF-IDF
    algorithm='brute',       # Mejor con cosine
    weights='uniform',      # Mejora precisión
    n_jobs=-1
)
knn.fit(X_train, y_train)

# Guardar el modelo entrenado en la misma carpeta
joblib.dump(knn, os.path.join(os.path.dirname(__file__), 'knn_model.joblib'))



train_preds = knn.predict(X_train)
val_preds = knn.predict(X_val)
test_preds = knn.predict(X_test)

# Metrics
acc_train = accuracy_score(y_train, train_preds)
acc_val = accuracy_score(y_val, val_preds)
acc_test = accuracy_score(y_test, test_preds)
cm = confusion_matrix(y_test, test_preds, labels=["negative", "neutral", "positive"])
report = classification_report(y_test, test_preds, digits=3)

output_lines = []
output_lines.append("=== KNN (n_neighbors=7) ===\n")
output_lines.append(f"Train Accuracy: {acc_train:.4f}\n")
output_lines.append(f"Validation Accuracy: {acc_val:.4f}\n")
output_lines.append(f"Test Accuracy: {acc_test:.4f}\n\n")
output_lines.append("Confusion Matrix (Test):\n")
output_lines.append(str(cm) + "\n")
output_lines.append("Classification Report (Test):\n")
output_lines.append(report + "\n")

# Print
for line in output_lines:
    print(line, end="")

# Save to file
with open("output_knn.txt", "w", encoding="utf-8") as f:
    for line in output_lines:
        f.write(line)
