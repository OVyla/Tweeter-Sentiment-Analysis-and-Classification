import os
import sys
import joblib
import json
import numpy as np
import warnings
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.model_selection import cross_val_score

# ==========================================
# CONFIGURACIÓ BÀSICA
# ==========================================
warnings.filterwarnings('ignore')
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.insert(0, project_root)
os.chdir(project_root)

# Importar dades
try:
    from AnalizarLimpiarDividir.vector_representation import load_and_vectorize_splits
    print("Loading TF-IDF data...")
    # Assegura't que max_features quadra amb el que necessites
    data = load_and_vectorize_splits(method='TFIDF', max_features=160000)
    X_train = data['X_train']
    y_train = data['y_train']
except ImportError:
    sys.exit("Error: No s'ha pogut importar load_and_vectorize_splits")

# Directoris de sortida
models_dir = os.path.join(current_dir, 'best_models_optuna')
os.makedirs(models_dir, exist_ok=True)
results_file = os.path.join(current_dir, 'optuna_results.txt')
params_file = os.path.join(current_dir, 'best_hyperparameters.json')

# ==========================================
# SUBSAMPLING (Vital per velocitat del Tuning)
# ==========================================
# Utilitzem una mostra per decidir els hiperparàmetres ràpidament
SAMPLE_SIZE = 10000
if X_train.shape[0] > SAMPLE_SIZE:
    print(f"Subsampling a {SAMPLE_SIZE} mostres per a la cerca d'hiperparàmetres...")
    indices = np.random.choice(X_train.shape[0], SAMPLE_SIZE, replace=False)
    X_subset = X_train[indices]
    y_subset = y_train[indices]
else:
    X_subset = X_train
    y_subset = y_train

# ==========================================
# FUNCIÓ OBJECTIU (Lògica d'Optuna)
# ==========================================
def create_model_structure(trial, model_name):
    """Defineix l'estructura i l'espai de cerca."""
    
    # --- LOGISTIC REGRESSION ---
    if "logistic" in model_name:
        C = trial.suggest_float("C", 0.01, 100, log=True)
        solver = trial.suggest_categorical("solver", ["lbfgs", "saga"])
        
        # Base settings
        lr_kwargs = {'C': C, 'solver': solver, 'max_iter': 500, 'n_jobs': -1}
        
        if "standard" in model_name:
            return LogisticRegression(multi_class='multinomial', **lr_kwargs)
        elif "ovr" in model_name:
            return LogisticRegression(multi_class='ovr', **lr_kwargs)
        elif "ovo" in model_name:
            # OvO no admet n_jobs dins del base estimator si el wrapper ja en té
            lr_kwargs['n_jobs'] = 1 
            return OneVsOneClassifier(LogisticRegression(**lr_kwargs), n_jobs=-1)

    # --- SVM (Molt lent) ---
    elif "svm" in model_name:
        C = trial.suggest_float("C", 0.1, 50, log=True)
        kernel = trial.suggest_categorical("kernel", ["linear", "rbf"])
        
        # Limitem max_iter per evitar bloquejos eterns durant el tuning
        svc_kwargs = {'C': C, 'kernel': kernel, 'max_iter': 1000}
        
        if "standard" in model_name:
            return SVC(**svc_kwargs)
        elif "ovr" in model_name:
            # IMPORTANT: n_jobs=1 per evitar OOM (Out of Memory)
            return OneVsRestClassifier(SVC(**svc_kwargs), n_jobs=1)
        elif "ovo" in model_name:
            # IMPORTANT: n_jobs=1 per evitar OOM
            return OneVsOneClassifier(SVC(**svc_kwargs), n_jobs=1)

    # --- RANDOM FOREST ---
    elif "random_forest" in model_name:
        n_estimators = trial.suggest_int("n_estimators", 50, 300)
        max_depth = trial.suggest_int("max_depth", 10, 50)
        
        rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
        return OneVsRestClassifier(rf, n_jobs=-1)

    return None

def objective(trial):
    model_name = trial.study.user_attrs["model_name"]
    model = create_model_structure(trial, model_name)
    
    # Cross-validation ràpid (3 folds)
    # IMPORTANT: n_jobs=1 aquí és vital. Si poses -1, multipliques la RAM per 3 (folds).
    scores = cross_val_score(model, X_subset, y_subset, cv=3, scoring='accuracy', n_jobs=1)
    return scores.mean()

# ==========================================
# EXECUCIÓ DEL BUCLE
# ==========================================
model_list = [
    "logistic_standard", "logistic_ovr", "logistic_ovo",
    "svm_standard", "svm_ovr", "svm_ovo",
    "random_forest_ovr"
]

all_best_params = {}

with open(results_file, 'w') as f:
    f.write("OPTUNA OPTIMIZATION RESULTS\n===========================\n")

for name in model_list:
    print(f"\n🚀 Optimizing: {name}...")
    
    # 1. OPTIMITZACIÓ (Amb mostra petita)
    study = optuna.create_study(direction="maximize")
    study.set_user_attr("model_name", name)
    study.optimize(objective, n_trials=20) # 20 proves
    
    best_params = study.best_params
    all_best_params[name] = best_params
    
    print(f"✅ Best params for {name}: {best_params}")
    print(f"   Best Subset CV Score: {study.best_value:.4f}")
    
    with open(results_file, 'a') as f:
        f.write(f"\nModel: {name}\n")
        f.write(f"Best CV Score (Subset): {study.best_value:.4f}\n")
        f.write(f"Params: {best_params}\n")

# Guardar tots els millors paràmetres en un JSON per al següent script
with open(params_file, 'w') as f:
    json.dump(all_best_params, f, indent=4)

print(f"\n💾 Hyperparameters saved to {params_file}")
print(f"🏁 Tuning finalitzat. Ara executa 'train_final_models.py' per entrenar amb tot el dataset.")