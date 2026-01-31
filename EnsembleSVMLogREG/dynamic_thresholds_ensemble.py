import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import sys, os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import sys, os
import warnings
warnings.filterwarnings('ignore')

import sys
import os

# --- INICI BLOC AUTO-CONFIGURACIÓ PATH ---
# Aquest codi puja nivells fins a trobar la carpeta 'preprocessing'
current_dir = os.path.dirname(os.path.abspath(__file__))
while current_dir != os.path.dirname(current_dir):  # Evita bucle infinit al root del sistema
    if os.path.exists(os.path.join(current_dir, 'preprocessing')):
        sys.path.append(os.path.join(current_dir, 'preprocessing'))
        break
    current_dir = os.path.dirname(current_dir)
# -----------------------------------------

import vector_representation as vr
project_root = current_dir

OUTPUT_FILE = os.path.join(current_dir, "dynamic_thresholds_ensemble_results.txt")

print("\n" + "="*80)
print("OPTIMIZACIÓN DE THRESHOLDS POR CLASE (VALIDACIÓN)")
print("="*80)

# 1. Cargar datasets
print("\n[1/4] Cargando datasets...")
datasets_dir = os.path.join(project_root, 'data', 'SPLIT')
train = pd.read_csv(os.path.join(datasets_dir, "twitter_trainBALANCED.csv"))
val = pd.read_csv(os.path.join(datasets_dir, "twitter_valBALANCED.csv"))
test = pd.read_csv(os.path.join(datasets_dir, "twitter_testBALANCED.csv"))

# 2. Cargar vectores TF-IDF
print("[2/4] Cargando vectores TF-IDF...")
vectors_path = os.path.join(project_root, 'data', 'VECTORS', 'tfidf')
X_train, X_val, X_test, _ = vr.load_tfidf(prefix=vectors_path)
y_val = val['label']
y_test = test['label']

print(f"  - Validation: {X_val.shape}")
print(f"  - Test: {X_test.shape}")

# 3. Cargar modelos base del cache
print("\n[3/4] Cargando modelos base desde caché...")
lr_model = joblib.load(os.path.join(current_dir, 'cache_lr_model.joblib'))
svm_model = joblib.load(os.path.join(current_dir, 'cache_svm_model.joblib'))
print("  ✓ LR y SVM cargados del caché")

# Weighted voting óptimo (según tu análisis)
lr_weight = 0.8
svm_weight = 0.2

# Probabilidades en validación y test
lr_proba_val = lr_model.predict_proba(X_val)
lr_proba_test = lr_model.predict_proba(X_test)
svm_proba_val = svm_model.predict_proba(X_val)
svm_proba_test = svm_model.predict_proba(X_test)

# Ensemble soft voting
ensemble_proba_val = (lr_weight * lr_proba_val + svm_weight * svm_proba_val) / (lr_weight + svm_weight)
ensemble_proba_test = (lr_weight * lr_proba_test + svm_weight * svm_proba_test) / (lr_weight + svm_weight)
classes = lr_model.classes_

# Buscar thresholds óptimos por clase usando validación
print("\n[4/4] Buscando thresholds óptimos por clase...")


# Grid más fino y búsqueda independiente por clase
grid = np.arange(0.3, 0.71, 0.001)
best_thresholds = []
for i, c in enumerate(classes):
    best_t = 0.5
    best_acc = 0
    for t in grid:
        preds = []
        for row in ensemble_proba_val:
            # Si la probabilidad de la clase i supera el threshold, predice esa clase
            if row[i] >= t:
                preds.append(c)
            else:
                # Si no, predice la de mayor probabilidad
                preds.append(classes[np.argmax(row)])
        acc = accuracy_score(y_val, preds)
        if acc > best_acc:
            best_acc = acc
            best_t = t
    best_thresholds.append(best_t)
    print(f"  {c}: {best_t:.2f}")

print(f"\n✓ Thresholds óptimos encontrados (validación):")
for i, c in enumerate(classes):
    print(f"  {c}: {best_thresholds[i]:.2f}")

# Aplicar thresholds óptimos al test set
test_preds = []
for row in ensemble_proba_test:
    if row[0] >= best_thresholds[0]:
        test_preds.append(classes[0])
    elif row[1] >= best_thresholds[1]:
        test_preds.append(classes[1])
    elif row[2] >= best_thresholds[2]:
        test_preds.append(classes[2])
    else:
        test_preds.append(classes[np.argmax(row)])

acc_test = accuracy_score(y_test, test_preds)
print(f"\n✓ Test Accuracy (Dynamic Thresholds): {acc_test:.4f} ({acc_test*100:.2f}%)")

# Guardar resultados
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("="*80 + "\n")
    f.write("OPTIMIZACIÓN DE THRESHOLDS POR CLASE (VALIDACIÓN)\n")
    f.write("="*80 + "\n\n")
    f.write(f"Thresholds óptimos: {dict(zip(classes, best_thresholds))}\n")
    f.write(f"Test Accuracy: {acc_test:.4f} ({acc_test*100:.2f}%)\n\n")
    f.write(classification_report(y_test, test_preds))
    f.write(f"\nConfusion Matrix:\n{confusion_matrix(y_test, test_preds)}\n")

print(f"\n✓ Resultados guardados en: {OUTPUT_FILE}")
print("="*80 + "\n")
