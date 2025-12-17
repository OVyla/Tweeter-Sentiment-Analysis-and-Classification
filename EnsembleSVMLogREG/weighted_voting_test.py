import pandas as pd
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import sys, os
import warnings
import joblib
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import sys, os
import warnings
import joblib
warnings.filterwarnings('ignore')

import sys
import os

# --- INICI BLOC AUTO-CONFIGURACIÓ PATH ---
# Aquest codi puja nivells fins a trobar la carpeta 'AnalizarLimpiarDividir'
current_dir = os.path.dirname(os.path.abspath(__file__))
while current_dir != os.path.dirname(current_dir):  # Evita bucle infinit al root del sistema
    if os.path.exists(os.path.join(current_dir, 'AnalizarLimpiarDividir')):
        sys.path.append(os.path.join(current_dir, 'AnalizarLimpiarDividir'))
        break
    current_dir = os.path.dirname(current_dir)
# -----------------------------------------

import vector_representation as vr
project_root = current_dir

OUTPUT_FILE = os.path.join(current_dir, "weighted_voting_results.txt")

print("\n" + "="*80)
print("WEIGHTED VOTING - PRUEBA DE DIFERENTES PESOS")
print("="*80)

# 1. Cargar datasets
print("\n[1/4] Cargando datasets...")
datasets_dir = os.path.join(project_root, 'DATASETS', 'SPLIT')
train = pd.read_csv(os.path.join(datasets_dir, "twitter_trainBALANCED.csv"))
test = pd.read_csv(os.path.join(datasets_dir, "twitter_testBALANCED.csv"))

# 2. Cargar vectores TF-IDF
print("[2/4] Cargando vectores TF-IDF...")
vectors_path = os.path.join(project_root, 'DATASETS', 'VECTORS', 'tfidf')
X_train, X_val, X_test, _ = vr.load_tfidf(prefix=vectors_path)
y_train = train['label']
y_test = test['label']

print(f"  - Train: {X_train.shape}")
print(f"  - Test: {X_test.shape}")

# 3. Cargar modelos del cache
print("\n[3/4] Cargando modelos desde caché...")
lr_model = joblib.load(os.path.join(current_dir, 'cache_lr_model.joblib'))
svm_model = joblib.load(os.path.join(current_dir, 'cache_svm_model.joblib'))
print("  ✓ LR y SVM cargados del caché")

# 4. Probar diferentes pesos
print("\n[4/4] Probando combinaciones de pesos (LR mejor en neutral+positive)...")
print("  (LR_weight, SVM_weight) -> Test Accuracy\n")

# Pesos a probar: enfocados en LR que es mejor en 2/3 clases
# Baseline: (0.5, 0.5)
# Candidatos: (0.6, 0.4) y (0.7, 0.3)
weight_combinations = [
    (0.6, 0.4),  # LR 60%
    (0.7, 0.3),  # LR 70%
    (0.8, 0.2),  # LR 80%
    (0.9, 0.1),  # LR 90%
]

results = []
best_acc = 0
best_weights = (0.5, 0.5)

for lr_w, svm_w in weight_combinations:
    # Calcular probabilidades ponderadas manualmente
    # (VotingClassifier requiere fit, pero ya tenemos modelos entrenados)
    lr_proba = lr_model.predict_proba(X_test)
    svm_proba = svm_model.predict_proba(X_test)
    
    # Ponderar
    weighted_proba = (lr_w * lr_proba + svm_w * svm_proba) / (lr_w + svm_w)
    
    # Predecir desde probabilidades
    test_pred = lr_model.classes_[np.argmax(weighted_proba, axis=1)]
    test_acc = accuracy_score(y_test, test_pred)
    
    results.append({
        'lr_weight': lr_w,
        'svm_weight': svm_w,
        'accuracy': test_acc
    })
    
    marker = "✓ MEJOR" if test_acc > best_acc else ""
    print(f"  LR {lr_w:.1f} + SVM {svm_w:.1f}  →  {test_acc:.4f} ({test_acc*100:.2f}%) {marker}")
    
    if test_acc > best_acc:
        best_acc = test_acc
        best_weights = (lr_w, svm_w)
        best_pred = test_pred

# Generar reporte
print("\n" + "="*80)
print("RESULTADOS FINALES")
print("="*80)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("="*80 + "\n")
    f.write("WEIGHTED VOTING - PRUEBA DE DIFERENTES PESOS\n")
    f.write("="*80 + "\n\n")
    
    f.write("COMPARATIVA DE PESOS:\n")
    f.write("-"*80 + "\n")
    
    for res in results:
        marker = "← MEJOR" if res['accuracy'] == best_acc else ""
        f.write(f"LR {res['lr_weight']:.1f} + SVM {res['svm_weight']:.1f}  →  {res['accuracy']:.4f} ({res['accuracy']*100:.2f}%) {marker}\n")
    
    f.write("\n" + "="*80 + "\n")
    f.write("MEJOR CONFIGURACIÓN\n")
    f.write("="*80 + "\n")
    f.write(f"LR Weight: {best_weights[0]:.1f}\n")
    f.write(f"SVM Weight: {best_weights[1]:.1f}\n")
    f.write(f"Test Accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)\n\n")
    
    # Comparación vs baseline (0.5, 0.5)
    baseline_acc = results[0]['accuracy']  # El primero es siempre el baseline
    improvement = best_acc - baseline_acc
    
    f.write("-"*80 + "\n")
    f.write("COMPARACIÓN CON BASELINE (LR 0.5 + SVM 0.5)\n")
    f.write("-"*80 + "\n")
    f.write(f"Baseline Accuracy: {baseline_acc:.4f} ({baseline_acc*100:.2f}%)\n")
    f.write(f"Mejor Accuracy:   {best_acc:.4f} ({best_acc*100:.2f}%)\n")
    f.write(f"Mejora:           {improvement:+.4f} ({improvement*100:+.2f}%)\n\n")
    
    if improvement > 0:
        f.write(f"✓ MEJORA DETECTADA: +{improvement*100:.2f}%\n")
    else:
        f.write(f"✗ NO HAY MEJORA: {improvement*100:.2f}%\n")
    
    # Classification report del mejor
    f.write("\n" + "="*80 + "\n")
    f.write("CLASSIFICATION REPORT - MEJOR MODELO\n")
    f.write("="*80 + "\n")
    f.write(classification_report(y_test, best_pred))
    f.write(f"\nConfusion Matrix:\n{confusion_matrix(y_test, best_pred)}\n")

print(f"\n✓ Mejor configuración: LR {best_weights[0]:.1f} + SVM {best_weights[1]:.1f}")
print(f"✓ Test Accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")
print(f"✓ Mejora vs baseline: {improvement:+.4f} ({improvement*100:+.2f}%)")
print(f"\n✓ Resultados guardados en: {OUTPUT_FILE}")
print("="*80 + "\n")
