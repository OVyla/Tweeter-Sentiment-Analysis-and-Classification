import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.preprocessing import label_binarize


# --- Cargar splits ---
train = pd.read_csv("twitter_trainBALANCED.csv")
val = pd.read_csv("twitter_valBALANCED.csv")
test = pd.read_csv("twitter_testBALANCED.csv")

# --- Vectorización TF-IDF mejorada ---
vectorizer = TfidfVectorizer(max_features=80000, ngram_range=(1,3), sublinear_tf=True,min_df=2,max_df=0.95 )
X_train = vectorizer.fit_transform(train['text'])
X_val = vectorizer.transform(val['text'])
X_test = vectorizer.transform(test['text'])

y_train = train['label']
y_val = val['label']
y_test = test['label']

# --- Entrenamiento y evaluación directa (rápido) ---
#logreg = LogisticRegression(C=2, solver='liblinear', class_weight='balanced', max_iter=300, random_state=42)

"""
# --- OneVsOneClassifier ---
from sklearn.multiclass import OneVsOneClassifier
logreg = OneVsOneClassifier(
    LogisticRegression(
        solver='liblinear',   # liblinear funciona bé per binari
        penalty='l2',
        C=2,
        max_iter=1000,
        random_state=42
    )
)

# --- MultinomialClassifier ---
logreg = LogisticRegression(
    C=2,
    solver='saga',
    penalty='l2',
    class_weight='balanced',  # o None si realment el dataset ja és molt equilibrat
    max_iter=1000,
    n_jobs=-1,
    random_state=42
)
"""

# --- OneVsRestClassifier ---
from sklearn.multiclass import OneVsRestClassifier
logreg = OneVsRestClassifier(
    LogisticRegression(
        solver='liblinear',
        penalty='l2',
        C=2,
        max_iter=1000,
        random_state=42
    )
)

logreg.fit(X_train, y_train)

# --- Evaluación en train ---
train_pred = logreg.predict(X_train)
print("\n--- TRAIN ---")
print(classification_report(y_train, train_pred))
print("Accuracy:", accuracy_score(y_train, train_pred))

# --- Validación ---
val_pred = logreg.predict(X_val)
print("\n--- VALIDATION ---")
print(classification_report(y_val, val_pred))
print("Accuracy:", accuracy_score(y_val, val_pred))

# --- Test final ---
test_pred = logreg.predict(X_test)
print("\n--- TEST ---")
print(classification_report(y_test, test_pred))
print("Accuracy:", accuracy_score(y_test, test_pred))


"""
# --- Grid SearchCV  ---
from sklearn.model_selection import GridSearchCV
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

# Model base
base_logreg = LogisticRegression(
    solver='saga',
    penalty='l2',
    max_iter=1000,
    n_jobs=-1,
    random_state=42
)

# Petita graella d'hiperparàmetres
param_grid = {
    'C': [0.25, 0.5, 1, 2],
    'class_weight': [None, 'balanced']
}

@ignore_warnings(category=ConvergenceWarning)
def run_grid_search():
    gs = GridSearchCV(
        base_logreg,
        param_grid,
        cv=3,
        scoring='f1_macro',
        verbose=2,
        n_jobs=-1
    )
    gs.fit(X_train, y_train)
    return gs

print("\n--- GridSearchCV: buscant millors hiperparàmetres... ---")
gs = run_grid_search()
print("\nMillors hiperparàmetres:")
print(gs.best_params_)

best_logreg = gs.best_estimator_

# --- VALIDACIÓ ---
val_pred_gs = best_logreg.predict(X_val)
print("\n--- VALIDATION (GridSearchCV best) ---")
print(classification_report(y_val, val_pred_gs))
print("Accuracy:", accuracy_score(y_val, val_pred_gs))

# --- TEST ---
test_pred_gs = best_logreg.predict(X_test)
print("\n--- TEST (GridSearchCV best) ---")
print(classification_report(y_test, test_pred_gs))
print("Accuracy:", accuracy_score(y_test, test_pred_gs))
"""

# ============================
# Matriu de confusió (VALIDATION)
# ============================

classes = logreg.classes_  # ['negative', 'neutral', 'positive'] en el teu cas

cm = confusion_matrix(y_val, val_pred, labels=classes)
print("\n--- CONFUSION MATRIX (VALIDATION) ---")
print(cm)

plt.figure()
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(xticks_rotation=45)
plt.title("Confusion matrix - Validation")
plt.tight_layout()
plt.savefig("GRAFIQUES/logreg/confusion_matrix_val.png", dpi=300)
plt.close()
print("Matriu de confusió guardada a 'GRAFIQUES/logreg/confusion_matrix_val.png'")


# ============================
# ROC i Precision-Recall (VALIDATION, one-vs-rest)
# ============================

# Probabilitats per classe
y_val_proba = logreg.predict_proba(X_val)

# Binaritzem les etiquetes reals: shape (n_samples, n_classes)
y_val_bin = label_binarize(y_val, classes=classes)

# --- ROC curves one-vs-rest ---
plt.figure()
for i, cls in enumerate(classes):
    fpr, tpr, thresholds = roc_curve(y_val_bin[:, i], y_val_proba[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{cls} (AUC = {roc_auc:.3f})")

plt.plot([0, 1], [0, 1], "k--", label="Random")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC curves (Validation, one-vs-rest)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("GRAFIQUES/logreg/roc_ovr_val.png", dpi=300)
plt.close()
print("Corba ROC one-vs-rest guardada a 'GRAFIQUES/logreg/roc_ovr_val.png'")


# --- Precision-Recall curves one-vs-rest ---
plt.figure()
for i, cls in enumerate(classes):
    precision, recall, thresholds = precision_recall_curve(
        y_val_bin[:, i], y_val_proba[:, i]
    )
    ap = average_precision_score(y_val_bin[:, i], y_val_proba[:, i])
    plt.plot(recall, precision, label=f"{cls} (AP = {ap:.3f})")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall curves (Validation, one-vs-rest)")
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("GRAFIQUES/logreg/pr_ovr_val.png", dpi=300)
plt.close()
print("Corba Precision-Recall one-vs-rest guardada a 'GRAFIQUES/logreg/pr_ovr_val.png'")


# ============================
# Buscar un llindar òptim per a una classe concreta (exemple: 'positive')
# ============================

target_class = "positive"  # canvia-ho per 'negative' o 'neutral' si vols
if target_class not in classes:
    raise ValueError(f"Classe objectiu {target_class} no està a {classes}")

target_idx = np.where(classes == target_class)[0][0]

probs_target = y_val_proba[:, target_idx]
labels_target = y_val_bin[:, target_idx]  # 1 si és target_class, 0 si no

# Curva ROC per aquesta classe
fpr, tpr, roc_thresholds = roc_curve(labels_target, probs_target)

# Youden's J = TPR - FPR, per trobar llindar amb millor separació
j_scores = tpr - fpr
best_idx = np.argmax(j_scores)
best_thresh = roc_thresholds[best_idx]

print(f"\n[THRESHOLD] Classe objectiu: {target_class}")
print(f"  Millor llindar (Youden J): {best_thresh:.4f}")
print(f"  TPR (Recall): {tpr[best_idx]:.3f}, FPR: {fpr[best_idx]:.3f}")

# Predicció binària amb aquest llindar
y_val_pred_pos_custom = (probs_target >= best_thresh).astype(int)

print("\n[METRIQUES BINÀRIES amb llindar òptim per a "
      f"'{target_class}' (one-vs-rest a VALIDATION)]")
print("  Precision:", precision_score(labels_target, y_val_pred_pos_custom))
print("  Recall   :", recall_score(labels_target, y_val_pred_pos_custom))
print("  F1       :", f1_score(labels_target, y_val_pred_pos_custom))
