import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
import sys, os
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import vector_representation as vr

# Cargar dataset
print("Cargando dataset...")
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
train = pd.read_csv(os.path.join(base_dir, 'twitter_trainBALANCED.csv')).sample(frac=0.3, random_state=42)
test = pd.read_csv(os.path.join(base_dir, 'twitter_testBALANCED.csv')).sample(frac=0.3, random_state=42)

X_train_full, _, X_test_full, _ = vr.load_tfidf(prefix=os.path.join(base_dir, "VECTORES/tfidf"))
X_train = X_train_full[train.index]
X_test = X_test_full[test.index]
y_train = train['label']
y_test = test['label']

print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

# Grid de parámetros a buscar
param_grid = {
    'C': [0.001, 0.01, 0.1, 0.5, 1, 2, 10],
    'loss': ['hinge', 'squared_hinge'],
    'dual': [True, False],
    'tol': [1e-4, 1e-3, 1e-2],
    'max_iter': [1000, 2000, 5000],
    'fit_intercept': [True, False],
    'intercept_scaling': [1, 10],
    'multi_class': ['ovr', 'crammer_singer'],
}

total_combos = np.prod([len(v) for v in param_grid.values()])
print(f"\nRealizando búsqueda ALEATORIA de parámetros...")
print(f"Total de combinaciones posibles: {total_combos}")
print(f"Probando: 50 combinaciones aleatorias (inteligentes)")

# Randomized Search con validación cruzada
svm = LinearSVC(random_state=42, verbose=0)
random_search = RandomizedSearchCV(
    svm, 
    param_grid, 
    n_iter=50,  # 50 combinaciones aleatorias
    cv=2,  # 2-fold cross validation (más rápido)
    scoring='accuracy',
    n_jobs=-1,  # Usar todos los cores
    verbose=2,
    random_state=42
)

print("\nEntrenando Randomized Search...")
random_search.fit(X_train, y_train)

print(f"\n{'='*60}")
print(f"MEJOR COMBINACIÓN ENCONTRADA")
print(f"{'='*60}")
print(f"Parámetros: {random_search.best_params_}")
print(f"Score CV (validación cruzada): {random_search.best_score_:.4f}")

# Evaluar en test (30%)
best_model = random_search.best_estimator_
train_pred = best_model.predict(X_train)
test_pred = best_model.predict(X_test)

train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)

print(f"\nAccuracy en Train (30%): {train_acc:.4f}")
print(f"Accuracy en Test (30%): {test_acc:.4f}")

# Validación con dataset COMPLETO
print(f"\n{'='*80}")
print(f"VALIDACIÓN CON DATASET COMPLETO (100%)")
print(f"{'='*80}")

train_full = pd.read_csv(os.path.join(base_dir, 'twitter_trainBALANCED.csv'))
test_full = pd.read_csv(os.path.join(base_dir, 'twitter_testBALANCED.csv'))

X_train_full_all, _, X_test_full_all, _ = vr.load_tfidf(prefix=os.path.join(base_dir, "VECTORES/tfidf"))
X_train_full = X_train_full_all[train_full.index]
X_test_full = X_test_full_all[test_full.index]
y_train_full = train_full['label']
y_test_full = test_full['label']

print(f"Entrenando con dataset COMPLETO...")
best_model_full = LinearSVC(**random_search.best_params_, random_state=42, verbose=0)
best_model_full.fit(X_train_full, y_train_full)

train_pred_full = best_model_full.predict(X_train_full)
test_pred_full = best_model_full.predict(X_test_full)

train_acc_full = accuracy_score(y_train_full, train_pred_full)
test_acc_full = accuracy_score(y_test_full, test_pred_full)

print(f"Train Accuracy (100%): {train_acc_full:.4f}")
print(f"Test Accuracy (100%): {test_acc_full:.4f}")
print(f"{'='*80}\n")

# Guardar resultados
results_df = pd.DataFrame(random_search.cv_results_)
results_df.to_csv('svm_random_search_results.csv', index=False)
print("✓ Resultados del Random Search guardados en: svm_random_search_results.csv")

# Guardar los mejores parámetros en un archivo de texto
with open('svm_best_params_random.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("MEJORES PARÁMETROS SVM (BÚSQUEDA ALEATORIA)\n")
    f.write("="*80 + "\n\n")
    f.write("MEJORES PARÁMETROS:\n")
    for param, value in random_search.best_params_.items():
        f.write(f"  {param}: {value}\n")
    f.write(f"\nScore CV (30%): {random_search.best_score_:.4f}\n")
    f.write(f"Train Accuracy (30%): {train_acc:.4f}\n")
    f.write(f"Test Accuracy (30%): {test_acc:.4f}\n")
    f.write(f"\n--- VALIDACIÓN CON 100% ---\n")
    f.write(f"Train Accuracy (100%): {train_acc_full:.4f}\n")
    f.write(f"Test Accuracy (100%): {test_acc_full:.4f}\n")
    f.write("="*80 + "\n")

print("✓ Mejores parámetros guardados en: svm_best_params_random.txt")

# Top 10 combinaciones
print("\nTop 10 mejores combinaciones:")
print("-" * 80)
top_10 = results_df.nlargest(10, 'mean_test_score')[['param_C', 'param_loss', 'param_dual', 'param_tol', 'param_max_iter', 'param_fit_intercept', 'param_intercept_scaling', 'param_multi_class', 'mean_test_score', 'std_test_score']]
print(top_10.to_string())
print("\n✓ ¡Búsqueda completada!")
