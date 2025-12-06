import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import vector_representation as vr

# Load vectorized data
X_train, X_val, X_test, _ = vr.load_tfidf(prefix="./VECTORES/tfidf")


# Load labels and sample 20,000 rows from each set
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
train = pd.read_csv(os.path.join(base_dir, 'twitter_trainBALANCED.csv')).sample(frac=0.5, random_state=42)
val = pd.read_csv(os.path.join(base_dir, 'twitter_valBALANCED.csv')).sample(frac=0.5, random_state=42)
test = pd.read_csv(os.path.join(base_dir, 'twitter_testBALANCED.csv')).sample(frac=0.5, random_state=42)
y_train = train['label'].reset_index(drop=True)
y_val = val['label'].reset_index(drop=True)
y_test = test['label'].reset_index(drop=True)


# Select the same sample indices from the vectorized data
X_train = X_train[train.index]
X_val = X_val[val.index]
X_test = X_test[test.index]

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
