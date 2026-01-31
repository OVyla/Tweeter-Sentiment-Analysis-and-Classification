import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import sys, os
import time
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import sys, os
import time
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

try:
    from lightgbm import LGBMClassifier
except ImportError:
    print("ERROR: LightGBM no está instalado. Instala con: pip install lightgbm")
    sys.exit(1)

OUTPUT_FILE = os.path.join(current_dir, "lightgbm_random_search_results.txt")

print("\n" + "="*80)
print("LIGHTGBM RANDOM SEARCH - HIPERPARAMETER TUNING")
print("="*80)

# Cargar datos
print("\n[PASO 1-2/5] Cargando datos y vectores...")
data = vr.load_and_vectorize_splits(method='TFIDF')
X_train = data['X_train']
X_val = data['X_val']
X_test = data['X_test']
y_train = pd.Series(data['y_train'])
y_val = pd.Series(data['y_val'])
y_test = pd.Series(data['y_test'])
print("  ✓ Datos cargados")

# Combinar train + val para modelo FINAL
from scipy.sparse import vstack
print("\n[PASO 3/5] Preparando datasets...")
X_full = vstack([X_train, X_val])
y_full = pd.concat([y_train, y_val], ignore_index=True)

# Crear SUBSET PEQUEÑO para Random Search (20% de validation = ~27K)
subset_size = int(len(y_val) * 0.2)
indices = np.random.RandomState(42).choice(len(y_val), subset_size, replace=False)
X_subset = X_val[indices]
y_subset = y_val.iloc[indices]

print(f"  ✓ Train: {X_train.shape}")
print(f"  ✓ Val (completo): {X_val.shape}")
print(f"  ✓ Subset para tuning (20% de val): {X_subset.shape}")
print(f"  ✓ Full (train+val para modelo final): {X_full.shape}")
print(f"  ✓ Test: {X_test.shape}")

# Espacio de búsqueda
print("\n[PASO 4/5] Configurando Random Search...")
param_dist = {
    'n_estimators': [50, 100, 150, 200],
    'learning_rate': [0.01, 0.05, 0.1, 0.15],
    'num_leaves': [20, 31, 50, 100],
    'max_depth': [5, 7, 10, 15, 20],
    'lambda_l1': [0.0, 0.5, 1.0],
    'lambda_l2': [0.0, 0.5, 1.0],
    'min_child_samples': [10, 20, 30],
    'feature_fraction': [0.7, 0.8, 0.9, 1.0],
    'bagging_fraction': [0.7, 0.8, 0.9, 1.0],
}

print(f"  Parámetros a probar: {len(param_dist)}")
for param, values in param_dist.items():
    print(f"    - {param}: {values}")
print("  ✓ Configuración lista")

# Random Search en VALIDATION SET (pequeño, rápido)
print("\n[PASO 4B/5] FASE 1: Random Search en subset pequeño...")
print("  Entrenando 20 combinaciones x 2-fold CV = 40 modelos...")
print("  (estimado 5-7 minutos)")
start_time = time.time()

base_lgbm = LGBMClassifier(
    objective='multiclass',
    num_class=len(y_subset.unique()),
    random_state=42,
    verbose=-1,
    force_col_wise=True
)

random_search = RandomizedSearchCV(
    base_lgbm,
    param_distributions=param_dist,
    n_iter=20,
    cv=2,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

random_search.fit(X_subset, y_subset)

elapsed_time = time.time() - start_time
minutes = int(elapsed_time // 60)
seconds = elapsed_time % 60

print(f"  ✓ Random Search completado en {minutes} min {seconds:.1f} seg")

# Mejores parámetros
best_params = random_search.best_params_
best_cv_score = random_search.best_score_

print(f"\n  >> MEJORES PARÁMETROS ENCONTRADOS:")
for param, value in best_params.items():
    print(f"     {param}: {value}")
print(f"  >> CV Score en subset: {best_cv_score:.4f}")

# FASE 2: Entrenar modelo FINAL en train+val combinados con los mejores parámetros
print(f"\n[PASO 5/5] FASE 2: Entrenando modelo FINAL en train+val (1.2M muestras)...")
print(f"  (estimado 2-3 minutos)")
best_model = LGBMClassifier(
    objective='multiclass',
    num_class=len(y_full.unique()),
    random_state=42,
    verbose=-1,
    force_col_wise=True,
    **best_params
)

best_model.fit(X_full, y_full)

# Predicciones
print(f"  ✓ Modelo entrenado. Generando predicciones...")
train_pred = best_model.predict(X_train)
val_pred = best_model.predict(X_val)
test_pred = best_model.predict(X_test)
print(f"  ✓ Predicciones generadas")

train_acc = accuracy_score(y_train, train_pred)
val_acc = accuracy_score(y_val, val_pred)
test_acc = accuracy_score(y_test, test_pred)

print(f"\n>> RESULTADOS FINALES:")
print(f"   Train Accuracy: {train_acc:.4f}")
print(f"   Val Accuracy:   {val_acc:.4f}")
print(f"   Test Accuracy:  {test_acc:.4f} ⭐")

# Guardar en caché
print(f"\n[PASO 5B/5] Guardando mejor modelo en caché...")
import joblib
joblib.dump(best_model, 'cache_lightgbm_model.joblib')
print(f"  ✓ Modelo guardado: cache_lightgbm_model.joblib")

# Guardar resultados
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("="*80 + "\n")
    f.write("LIGHTGBM RANDOM SEARCH - HYPERPARAMETER TUNING\n")
    f.write("="*80 + "\n\n")
    
    f.write("ESTRATEGIA:\n")
    f.write("  Fase 1: Random Search en subset pequeño (27K = 20% val) - RAPIDO\n")
    f.write("  Fase 2: Entrena modelo FINAL en train+val (1.2M) con esos params\n\n")
    
    f.write(f"Tiempo total: {minutes} min {seconds:.1f} seg\n")
    f.write(f"Combinaciones probadas: 20 (en subset de 27K muestras)\n")
    f.write(f"Cross-validation folds: 2 (en subset)\n")
    f.write(f"Modelo final entrenado en: train+val combinados (1.2M muestras)\n\n")
    
    f.write("-"*80 + "\n")
    f.write("MEJORES PARAMETROS ENCONTRADOS (en subset de 27K)\n")
    f.write("-"*80 + "\n\n")
    
    for param, value in best_params.items():
        f.write(f"{param}: {value}\n")
    
    f.write(f"\nCV Score en subset: {best_cv_score:.4f}\n\n")
    
    f.write("-"*80 + "\n")
    f.write("ACCURACIES\n")
    f.write("-"*80 + "\n\n")
    
    f.write(f"Train: {train_acc:.4f}\n")
    f.write(f"Val:   {val_acc:.4f}\n")
    f.write(f"Test:  {test_acc:.4f}\n\n")
    
    f.write("-"*80 + "\n")
    f.write("TEST SET RESULTS\n")
    f.write("-"*80 + "\n\n")
    
    f.write("CONFUSION MATRIX:\n")
    f.write(str(confusion_matrix(y_test, test_pred)) + "\n\n")
    
    f.write("CLASSIFICATION REPORT:\n")
    f.write(classification_report(y_test, test_pred))
    
    f.write("\n" + "-"*80 + "\n")
    f.write("TOP 5 MEJORES CONFIGURACIONES PROBADAS\n")
    f.write("-"*80 + "\n\n")
    
    results_df = pd.DataFrame(random_search.cv_results_)
    results_df = results_df.sort_values('rank_test_score')
    
    for idx, row in results_df.head(5).iterrows():
        rank = row['rank_test_score']
        score = row['mean_test_score']
        std = row['std_test_score']
        
        f.write(f"\n#{int(rank)}: {score:.4f} (+/- {std:.4f})\n")
        params = {k.replace('param_', ''): v for k, v in row.items() if k.startswith('param_') and pd.notna(v)}
        for param, value in params.items():
            f.write(f"  {param}: {value}\n")

print(f"\n✓ Resultados guardados en {OUTPUT_FILE}")
print("="*80 + "\n")
