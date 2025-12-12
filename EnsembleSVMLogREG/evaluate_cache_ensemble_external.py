import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import sys, os
import warnings
warnings.filterwarnings('ignore')

# Añadir MODELOS al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import vector_representation as vr

OUTPUT_FILE = "cache_ensemble_external_clean_balanced_results.txt"

print("\n" + "="*80)
print("EVALUACIÓN DEL ENSEMBLE GUARDADO SOBRE external_clean_balanced.csv")
print("="*80)

# 1. Cargar external_clean_balanced.csv
print("\n[1/3] Cargando external_clean_balanced.csv...")
external = pd.read_csv("../../external_clean_balanced.csv")

# 2. Cargar vectores TF-IDF para external

print("[2/3] Cargando vectorizador TF-IDF y transformando external...")
# Asume que el archivo tiene una columna 'text' y 'label'

vectorizer = joblib.load("../../VECTORES/tfidf_vectorizer.pkl")
X_external = vectorizer.transform(external['text'])
y_external = external['label']

print(f"  - External: {X_external.shape}")

# 3. Cargar ensemble desde cache
print("\n[3/3] Cargando ensemble desde cache...")
ensemble = joblib.load('cache_ensemble_model.joblib')
print("  ✓ Ensemble cargado")

# Evaluar


print("\nEvaluando ensemble sobre external_clean_balanced.csv...")
pred = ensemble.predict(X_external)

# Contar neutrales en ground truth externo
n_neutral_external = (y_external == 'neutral').sum() if 'neutral' in y_external.unique() else 0
# Contar neutrales en predicción antes de reclasificar
n_neutral_pred = (pred == 'neutral').sum() if 'neutral' in ensemble.classes_ else 0

# Si hay predicciones 'neutral', reasignarlas a 'positive' o 'negative' según la mayor probabilidad
if hasattr(ensemble, 'predict_proba'):
    proba = ensemble.predict_proba(X_external)
    classes = ensemble.classes_
    idx_neutral = list(classes).index('neutral') if 'neutral' in classes else None
    idx_positive = list(classes).index('positive') if 'positive' in classes else None
    idx_negative = list(classes).index('negative') if 'negative' in classes else None
    pred_adj = []
    for i, p in enumerate(pred):
        if p == 'neutral' and idx_neutral is not None:
            prob_pos = proba[i, idx_positive] if idx_positive is not None else 0
            prob_neg = proba[i, idx_negative] if idx_negative is not None else 0
            if prob_pos >= prob_neg:
                reassigned = 'positive'
            else:
                reassigned = 'negative'
            pred_adj.append(reassigned)
        else:
            pred_adj.append(p)
    pred = pred_adj

acc = accuracy_score(y_external, pred)

print(f"\n✓ Accuracy: {acc:.4f} ({acc*100:.2f}%)")
# Proporción de neutrales en ground truth externo
if len(y_external) > 0:
    prop_neutral_external = n_neutral_external / len(y_external)
else:
    prop_neutral_external = 0
print(f"Neutrales en ground truth externo: {n_neutral_external} ({prop_neutral_external:.4%})")
# Proporción de neutrales en predicción antes de reclasificar
if len(pred) > 0:
    prop_neutral_pred = n_neutral_pred / len(pred)
else:
    prop_neutral_pred = 0
print(f"Neutrales en predicción (antes de reclasificar): {n_neutral_pred} ({prop_neutral_pred:.4%})")

# Guardar resultados

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("="*80 + "\n")
    f.write("EVALUACIÓN DEL ENSEMBLE GUARDADO SOBRE external_clean_balanced.csv\n")
    f.write("="*80 + "\n\n")
    f.write(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)\n\n")
    f.write(classification_report(y_external, pred))
    f.write(f"\nConfusion Matrix:\n{confusion_matrix(y_external, pred)}\n")
    f.write(f"\nNeutrales en ground truth externo: {n_neutral_external} ({prop_neutral_external:.4%})\n")
    f.write(f"Neutrales en predicción (antes de reclasificar): {n_neutral_pred} ({prop_neutral_pred:.4%})\n")

print(f"\n✓ Resultados guardados en: {OUTPUT_FILE}")
print("="*80 + "\n")
