
import pandas as pd
import numpy as np
import time
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import sys, os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import time
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
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

# ==========================================
# CONFIGURACIÓN
# ==========================================
OUTPUT_FILE = os.path.join(current_dir, "stacking_lr_svm_results.txt")
start_time = time.time()

print("\n" + "="*80)
print("STACKING - LOGISTIC REGRESSION OVR + SVM OVR")
print("="*80)

# 1. Cargar datasets
print("\n[1/4] Cargando datasets...")
# (Els datasets es carreguen internament a load_and_vectorize_splits, però mantenim la lectura original per si es fan servir raw dataframes, 
#  o confiem en load_and_vectorize_splits que retorna train_df també)

# 2. Cargar vectores TF-IDF
print("[2/4] Cargando vectores TF-IDF...")
data = vr.load_and_vectorize_splits(method='TFIDF')
X_train = data['X_train']
X_val   = data['X_val']
X_test  = data['X_test']
y_train = data['y_train']
y_val   = data['y_val']
y_test  = data['y_test']

print(f"  - Train: {X_train.shape}")
print(f"  - Validation: {X_val.shape}")
print(f"  - Test: {X_test.shape}")

# Usar DATASET COMPLETO (sin muestreo)
print("\n[FULL DATASET] Usando 100% del dataset para mejor aprendizaje...")

# 3. Crear modelos base con OneVsRestClassifier
print("\n[3/4] Creando modelos base con OneVsRestClassifier...")

# Logistic Regression OVR con liblinear
lr_base = LogisticRegression(
    C=2.0,
    max_iter=500,
    penalty='l2',
    class_weight='balanced',
    solver='liblinear',
    random_state=42
)
lr_model = OneVsRestClassifier(lr_base)
print("  ✓ Logistic Regression OVR (base)")

# SVM OVR
svm_base = LinearSVC(
    C=0.1,
    tol=0.0001,
    max_iter=1000,
    loss='squared_hinge',
    intercept_scaling=10,
    fit_intercept=True,
    dual=False,
    random_state=42
)
svm_ovr = OneVsRestClassifier(svm_base)
# Envolver con Calibration para predict_proba
svm_model = CalibratedClassifierCV(svm_ovr)
print("  ✓ SVM OVR (base)")

# Meta-learner: Logistic Regression OVR con liblinear
meta_lr_base = LogisticRegression(
    C=1.0,
    max_iter=500,
    penalty='l2',
    class_weight='balanced',
    solver='liblinear',
    random_state=42
)
meta_learner = OneVsRestClassifier(meta_lr_base)
print("  ✓ Meta-learner (Logistic Regression OVR)")

# 4. Crear Stacking Classifier
print("\n[4/4] Creando Stacking Classifier...")
print("  Configurando base learners + meta-learner...")

stacking = StackingClassifier(
    estimators=[
        ('logistic_regression', lr_model),
        ('svm', svm_model)
    ],
    final_estimator=meta_learner,
    cv=3  # 3-fold cross-validation para entrenar meta-learner
)

print("  Entrenando Stacking...")
stacking.fit(X_train, y_train)

print("  ✓ Stacking entrenado exitosamente")
# Guardar modelo stacking en caché justo después de entrenar

joblib.dump(stacking, os.path.join(current_dir, 'cache_stacking_model.joblib'))
print("\n✓ Modelo stacking guardado en cache_stacking_model.joblib")
# Guardar modelo stacking en caché justo después de entrenar
import joblib
joblib.dump(stacking, os.path.join(current_dir, 'cache_stacking_model.joblib'))
print("\n✓ Modelo stacking guardado en cache_stacking_model.joblib")

# Predicciones
print("\n[PREDICCIONES] Generando predicciones...")
print("  - Prediciendo en train set...")
train_pred = stacking.predict(X_train)
train_acc = accuracy_score(y_train, train_pred)
print(f"    ✓ Train Accuracy: {train_acc:.4f}")

print("  - Prediciendo en validation set...")
val_pred = stacking.predict(X_val)
val_acc = accuracy_score(y_val, val_pred)
print(f"    ✓ Validation Accuracy: {val_acc:.4f}")

print("  - Prediciendo en test set...")
test_pred = stacking.predict(X_test)
test_acc = accuracy_score(y_test, test_pred)
print(f"    ✓ Test Accuracy: {test_acc:.4f}")

# Calcular tiempo
end_time = time.time()
total_seconds = end_time - start_time
mins = int(total_seconds // 60)
secs = total_seconds % 60
time_str = f"{mins} min {secs:.2f} s"

# Guardar resultados
print("\nGuardando resultados...")
with open(OUTPUT_FILE, "w") as f:
    f.write("="*80 + "\n")
    f.write("STACKING - LOGISTIC REGRESSION + SVM\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"Tiempo total de ejecución: {time_str}\n")
    f.write(f"Meta-learner: Logistic Regression\n")
    f.write(f"Cross-validation folds: 3\n")
    
    f.write("\n" + "-"*80 + "\n")
    f.write("CONFIGURACIÓN DE MODELOS BASE\n")
    f.write("-"*80 + "\n")
    
    f.write("\n1. LOGISTIC REGRESSION (Base Learner)\n")
    f.write("  C: 2.0\n")
    f.write("  max_iter: 500\n")
    f.write("  penalty: l2\n")
    f.write("  class_weight: balanced\n")
    f.write("  multi_class: ovr\n")
    f.write("  solver: lbfgs\n")
    
    f.write("\n2. LINEAR SVM (Base Learner)\n")
    f.write("  C: 0.1\n")
    f.write("  tol: 0.0001\n")
    f.write("  multi_class: ovr\n")
    f.write("  max_iter: 1000\n")
    f.write("  loss: squared_hinge\n")
    f.write("  intercept_scaling: 10\n")
    f.write("  fit_intercept: True\n")
    f.write("  dual: False\n")
    
    f.write("\n3. META-LEARNER (Logistic Regression)\n")
    f.write("  C: 1.0\n")
    f.write("  max_iter: 500\n")
    f.write("  penalty: l2\n")
    f.write("  class_weight: balanced\n")
    f.write("  multi_class: ovr\n")
    f.write("  solver: lbfgs\n")
    
    f.write("\n" + "-"*80 + "\n")
    f.write("RESULTADOS STACKING\n")
    f.write("-"*80 + "\n")
    f.write(f"Train Accuracy: {train_acc:.4f}\n")
    f.write(f"Validation Accuracy: {val_acc:.4f}\n")
    f.write(f"Test Accuracy: {test_acc:.4f}\n")
    
    f.write("\n" + "-"*80 + "\n")
    f.write("TRAIN CLASSIFICATION REPORT\n")
    f.write("-"*80 + "\n")
    f.write(classification_report(y_train, train_pred))
    f.write(f"\nConfusion Matrix:\n{confusion_matrix(y_train, train_pred)}\n")
    
    f.write("\n" + "-"*80 + "\n")
    f.write("VALIDATION CLASSIFICATION REPORT\n")
    f.write("-"*80 + "\n")
    f.write(classification_report(y_val, val_pred))
    f.write(f"\nConfusion Matrix:\n{confusion_matrix(y_val, val_pred)}\n")
    
    f.write("\n" + "-"*80 + "\n")
    f.write("TEST CLASSIFICATION REPORT\n")
    f.write("-"*80 + "\n")
    f.write(classification_report(y_test, test_pred))
    f.write(f"\nConfusion Matrix:\n{confusion_matrix(y_test, test_pred)}\n")
    
    f.write("\n" + "-"*80 + "\n")
    f.write("CÓMO FUNCIONA STACKING\n")
    f.write("-"*80 + "\n")
    f.write("1. Los 2 modelos base (LR + SVM) entrenan en el dataset completo\n")
    f.write("2. Durante cross-validation (3-fold), cada fold se usa para entrenar el meta-learner\n")
    f.write("3. El meta-learner aprende a COMBINAR optimalmente las predicciones de LR + SVM\n")
    f.write("4. En predicción, ambos modelos base predicen, y el meta-learner decide la clase final\n")
    f.write("\nVentaja sobre Voting:\n")
    f.write("  - El meta-learner aprende cuándo confiar en LR vs SVM\n")
    f.write("  - Típicamente da 1-2% más accuracy que soft voting\n")

print(f"\n✓ Resultados guardados en: {OUTPUT_FILE}")

print("\n" + "="*80)
print("RESUMEN FINAL")
print("="*80)
print(f"Train Accuracy: {train_acc:.4f}")
print(f"Validation Accuracy: {val_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Tiempo total: {time_str}")
print("="*80 + "\n")

print("\nCOMPARATIVA CON MODELOS INDIVIDUALES:")
print("-" * 80)
print(f"SVM (individual):              79.99%")
print(f"Logistic Regression (indiv):   79.02%")
print(f"Ensemble Soft Voting (LR+SVM): 80.26%")
print(f"STACKING (LR+SVM):             {test_acc:.4f} ⭐")
print(f"Random Forest (indiv):         72.64%")
print(f"Bernoulli Naive Bayes (indiv): 74.65%")
print(f"KNN (indiv):                   62.31%")
print("-" * 80 + "\n")
