import os
import sys
import joblib
import json
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

# ==========================================
# CONFIGURACIÓ BÀSICA
# ==========================================
warnings.filterwarnings('ignore')
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.insert(0, project_root)
os.chdir(project_root)

# Importar dades COMPLETES (Sense Subsampling)
try:
    from AnalizarLimpiarDividir.vector_representation import load_and_vectorize_splits
    print("Loading FULL TF-IDF data (No Subsampling)...")
    # Carreguem tot el dataset
    data = load_and_vectorize_splits(method='TFIDF', max_features=160000)
    X_train = data['X_train']
    y_train = data['y_train']
except ImportError:
    sys.exit("Error: No s'ha pogut importar load_and_vectorize_splits")

# Directoris
models_dir = os.path.join(current_dir, 'best_models_optuna')
os.makedirs(models_dir, exist_ok=True)
params_file = os.path.join(current_dir, 'best_hyperparameters.json')

if not os.path.exists(params_file):
    sys.exit(f"Error: No s'ha trobat {params_file}. Executa primer run_hyperparameter_tuning.py")

# Carregar hiperparàmetres
with open(params_file, 'r') as f:
    all_best_params = json.load(f)

print(f"Loaded hyperparameters for {len(all_best_params)} models.")

# ==========================================
# BUCLE D'ENTRENAMENT FINAL
# ==========================================
for name, params in all_best_params.items():
    print(f"\n🔄 Training {name} on FULL dataset...")
    
    final_model = None
    
    # Reconstrucció del model amb els paràmetres carregats
    if "logistic" in name:
        lr_args = {'C': params['C'], 'solver': params['solver'], 'max_iter': 2000, 'n_jobs': -1}
        if "standard" in name:
            final_model = LogisticRegression(multi_class='multinomial', **lr_args)
        elif "ovr" in name:
            final_model = LogisticRegression(multi_class='ovr', **lr_args)
        elif "ovo" in name:
            lr_args['n_jobs'] = 1 
            final_model = OneVsOneClassifier(LogisticRegression(**lr_args), n_jobs=-1)
            
    elif "svm" in name:
        svc_args = {'C': params['C'], 'kernel': params['kernel'], 'max_iter': 2000}
        if "standard" in name:
            final_model = SVC(**svc_args)
        elif "ovr" in name:
            # Aquí podem intentar n_jobs=-1 si tens RAM, sinó posa 1
            final_model = OneVsRestClassifier(SVC(**svc_args), n_jobs=-1)
        elif "ovo" in name:
            final_model = OneVsOneClassifier(SVC(**svc_args), n_jobs=-1)
            
    elif "random_forest" in name:
        rf = RandomForestClassifier(n_estimators=params['n_estimators'], 
                                    max_depth=params['max_depth'], 
                                    random_state=42, n_jobs=-1)
        final_model = OneVsRestClassifier(rf, n_jobs=-1)

    # Entrenar i Guardar
    if final_model:
        try:
            final_model.fit(X_train, y_train)
            save_path = os.path.join(models_dir, f"{name}_best_full.joblib")
            joblib.dump(final_model, save_path)
            print(f"💾 Saved final model to: {save_path}")
        except Exception as e:
            print(f"❌ Error training {name}: {e}")

print("\n🏁 Tots els models han estat processats.")
