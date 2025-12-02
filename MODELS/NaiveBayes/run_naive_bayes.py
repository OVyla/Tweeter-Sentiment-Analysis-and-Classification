

import joblib
import numpy as np
from naive_bayes import model_complement, model_grid_search
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import sys, os
# Añadir MODELOS al path para importar vector_representation
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import vector_representation as vr

# Cargar datos vectorizados con la función load_tfidf
X_train, X_val, X_test, _ = vr.load_tfidf(prefix="./VECTORES/tfidf")

# Cargar etiquetas directamente de los CSV
train = pd.read_csv("./twitter_trainBALANCED.csv")
val = pd.read_csv("./twitter_valBALANCED.csv")
test = pd.read_csv("./twitter_testBALANCED.csv")
y_train = train['label']
y_val = val['label']
y_test = test['label']



# Usar grid search para MultinomialNB


model_complement = model_grid_search(X_train, y_train, model_type='bernoulli')
train_preds_comp = model_complement.predict(X_train)
val_preds_comp = model_complement.predict(X_val)
test_preds_comp = model_complement.predict(X_test)
acc_train_comp = accuracy_score(y_train, train_preds_comp)
acc_val_comp = accuracy_score(y_val, val_preds_comp)
acc_test_comp = accuracy_score(y_test, test_preds_comp)
cm_comp = confusion_matrix(y_test, test_preds_comp, labels=["negative", "neutral", "positive"])
report_comp = classification_report(y_test, test_preds_comp, digits=3)

output_lines = []
output_lines.append("=== NAIVE BAYES (Bernoulli, alpha=1.0) ===\n")
output_lines.append(f"Train Accuracy: {acc_train_comp:.4f}\n")
output_lines.append(f"Validation Accuracy: {acc_val_comp:.4f}\n")
output_lines.append(f"Test Accuracy: {acc_test_comp:.4f}\n\n")
output_lines.append("Confusion Matrix (Test):\n")
output_lines.append(str(cm_comp) + "\n")
output_lines.append("Classification Report (Test):\n")
output_lines.append(report_comp + "\n")

# Imprimir por pantalla
for line in output_lines:
	print(line, end="")

# Guardar en archivo output.txt en la misma carpeta
with open("output.txt", "w", encoding="utf-8") as f:
	for line in output_lines:
		f.write(line)
