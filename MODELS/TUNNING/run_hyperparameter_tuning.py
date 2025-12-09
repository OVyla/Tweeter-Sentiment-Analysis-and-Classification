import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

# ==========================================
# SETUP PATHS
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.insert(0, project_root)

try:
    from AnalizarLimpiarDividir.vector_representation import load_and_vectorize_splits
except ImportError:
    print("Error: Could not import 'load_and_vectorize_splits'. Check your directory structure.")
    sys.exit(1)

# Output directories
output_dir = current_dir
models_dir = os.path.join(output_dir, 'best_models')
os.makedirs(models_dir, exist_ok=True)
results_file = os.path.join(output_dir, 'tuning_results.txt')

# ==========================================
# LOAD DATA
# ==========================================
print("Loading TF-IDF data...")
data = load_and_vectorize_splits(method='TFIDF')
X_train_full = data['X_train']
y_train_full = data['y_train']

# ==========================================
# SUBSET DATA FOR TUNING
# ==========================================
# Tuning on the full dataset is too slow. We use a representative subset.
SAMPLE_SIZE = 20000 
if X_train_full.shape[0] > SAMPLE_SIZE:
    print(f"Subsampling training data to {SAMPLE_SIZE} samples for hyperparameter tuning...")
    # Use pandas sampling if y_train is a Series, else numpy
    if hasattr(y_train_full, 'sample'):
        y_subset = y_train_full.sample(n=SAMPLE_SIZE, random_state=42)
        X_subset = X_train_full[y_subset.index]
    else:
        indices = np.random.choice(X_train_full.shape[0], SAMPLE_SIZE, replace=False)
        X_subset = X_train_full[indices]
        y_subset = y_train_full[indices]
else:
    X_subset = X_train_full
    y_subset = y_train_full

# ==========================================
# DEFINE MODELS AND GRIDS
# ==========================================
# Note: For wrapped classifiers (OvR, OvO), parameters usually need 'estimator__' prefix
# but sklearn's GridSearchCV can sometimes handle it if passed directly to the wrapper constructor 
# or via the specific syntax. Here we define the base estimator with params where possible.

configs = [
    {
        "name": "logistic_standard_tfidf",
        "model": LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000),
        "params": {
            'C': [0.1, 1, 10],
            'solver': ['lbfgs', 'saga']
        }
    },
    {
        "name": "logistic_ovr_tfidf",
        "model": LogisticRegression(multi_class='ovr', solver='lbfgs', max_iter=1000),
        "params": {
            'C': [0.1, 1, 10]
        }
    },
    {
        "name": "logistic_ovo_tfidf",
        "model": OneVsOneClassifier(LogisticRegression(solver='lbfgs', max_iter=1000)),
        "params": {
            'estimator__C': [0.1, 1, 10]
        }
    },
    {
        "name": "svm_standard_tfidf",
        "model": SVC(max_iter=2000), # Limit iter to prevent hanging
        "params": {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf']
        }
    },
    {
        "name": "svm_ovr_tfidf",
        "model": OneVsRestClassifier(SVC(max_iter=2000)),
        "params": {
            'estimator__C': [0.1, 1, 10],
            'estimator__kernel': ['linear', 'rbf']
        }
    },
    {
        "name": "svm_ovo_tfidf",
        "model": OneVsOneClassifier(SVC(max_iter=2000)),
        "params": {
            'estimator__C': [0.1, 1, 10],
            'estimator__kernel': ['linear', 'rbf']
        }
    },
    {
        "name": "random_forest_ovr",
        "model": OneVsRestClassifier(RandomForestClassifier(random_state=42)),
        "params": {
            'estimator__n_estimators': [50, 100],
            'estimator__max_depth': [10, 20, None]
        }
    }
]

# ==========================================
# EXECUTE TUNING
# ==========================================
with open(results_file, 'w') as f:
    f.write("HYPERPARAMETER TUNING RESULTS\n")
    f.write("=============================\n\n")

print(f"Starting tuning for {len(configs)} models...")

for config in configs:
    name = config['name']
    model = config['model']
    params = config['params']
    
    print(f"\n--- Tuning {name} ---")
    print(f"Grid: {params}")
    
    try:
        grid = GridSearchCV(model, params, cv=3, scoring='accuracy', n_jobs=6, verbose=1)
        grid.fit(X_subset, y_subset)
        
        best_score = grid.best_score_
        best_params = grid.best_params_
        
        print(f"Best Score: {best_score:.4f}")
        print(f"Best Params: {best_params}")
        
        # Save results to text file
        with open(results_file, 'a') as f:
            f.write(f"Model: {name}\n")
            f.write(f"Best Accuracy (CV): {best_score:.4f}\n")
            f.write(f"Best Parameters: {best_params}\n")
            f.write("-" * 40 + "\n")
            
        # Save best model
        model_path = os.path.join(models_dir, f"{name}_best.joblib")
        joblib.dump(grid.best_estimator_, model_path)
        print(f"Saved best model to {model_path}")
        
    except Exception as e:
        print(f"Error tuning {name}: {e}")
        with open(results_file, 'a') as f:
            f.write(f"Model: {name} - FAILED: {e}\n")
            f.write("-" * 40 + "\n")

print(f"\nTuning complete. Results saved to {results_file}")
