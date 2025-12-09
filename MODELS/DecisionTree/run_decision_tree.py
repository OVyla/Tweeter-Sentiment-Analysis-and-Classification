import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import joblib

# Añadir la ruta al vector_representation si es necesario
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.insert(0, project_root)

from AnalizarLimpiarDividir.vector_representation import load_and_vectorize_splits
from MODELS.DecisionTree.decision_tree_model import train_decision_tree

# Directorios de salida
graphs_dir = os.path.join(project_root, 'GRAFIQUES', 'DecisionTree')
os.makedirs(graphs_dir, exist_ok=True)

output_dir = os.path.join(project_root, 'MODELS', 'DecisionTree')
os.makedirs(output_dir, exist_ok=True)

# Cargar datos vectorizados (TF-IDF)
data = load_and_vectorize_splits(method='TFIDF')
X_train = data["X_train"]
X_val   = data["X_val"]
X_test  = data["X_test"]
y_train = data["y_train"]
y_val   = data["y_val"]
y_test  = data["y_test"]

OUTPUT_FILE = os.path.join(output_dir, "output_decision_tree.txt")

def save_report(f, model_name, title, y_true, y_pred):
    report = f"\n=== {model_name} ===\n"
    report += f"--- {title} ---\n"
    report += f"Accuracy: {accuracy_score(y_true, y_pred):.4f}\n"
    report += f"Confusion Matrix:\n{confusion_matrix(y_true, y_pred)}\n"
    report += classification_report(y_true, y_pred)
    report += "-" * 60 + "\n"
    
    f.write(report)
    print(report)

def plot_roc_curve(model, X, y, class_names, title, output_path):
    # Binarizar las etiquetas
    from sklearn.preprocessing import label_binarize
    y_bin = label_binarize(y, classes=class_names)
    y_score = model.predict_proba(X)
    n_classes = len(class_names)
    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'ROC {class_names[i]} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {title}')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_precision_recall(model, X, y, class_names, title, output_path):
    from sklearn.preprocessing import label_binarize
    y_bin = label_binarize(y, classes=class_names)
    y_score = model.predict_proba(X)
    n_classes = len(class_names)
    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_bin[:, i], y_score[:, i])
        plt.plot(recall, precision, lw=2, label=f'{class_names[i]}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {title}')
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# Entrenar Decision Tree
model = train_decision_tree(
    X_train, y_train,
    max_depth=20,           # Valores originales
    min_samples_leaf=10,    # Valores originales
    random_state=42
)

# Save the model so it can be picked up by the benchmark generator
# We append _TFIDF so the generator knows which vectorizer to use
model_path = os.path.join(output_dir, "decision_tree_TFIDF.joblib")
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("RESUMEN DECISION TREE\n")
    f.write("=" * 40 + "\n")

    class_names = np.unique(y_train)
    for title, X, y in [
        ("TRAIN", X_train, y_train),
        ("VALIDATION", X_val, y_val),
        ("TEST", X_test, y_test)
    ]:
        y_pred = model.predict(X)
        save_report(f, "Decision Tree", title, y, y_pred)
        # Graficar ROC y Precision-Recall solo para VALIDATION y TEST
        if title in ("VALIDATION", "TEST"):
            roc_path = os.path.join(graphs_dir, f"roc_curve_{title.lower()}.png")
            pr_path = os.path.join(graphs_dir, f"precision_recall_{title.lower()}.png")
            plot_roc_curve(model, X, y, class_names, title, roc_path)
            plot_precision_recall(model, X, y, class_names, title, pr_path)
            print(f"Guardada curva ROC en {roc_path} y Precision-Recall en {pr_path}")

print(f"Resultados guardados en {OUTPUT_FILE}")
