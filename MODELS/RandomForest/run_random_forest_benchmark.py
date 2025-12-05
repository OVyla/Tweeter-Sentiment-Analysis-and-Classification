# MODELS/run_random_forest_benchmark.py

import os
import sys
import time
import warnings
import contextlib

from sklearn.metrics import classification_report, accuracy_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from AnalizarLimpiarDividir.vector_representation import load_and_vectorize_splits
from MODELS.RandomForest.random_forest_model import (
    lightgbm_multiclass,
    lightgbm_ovr,
    ada_boost_ovr,
    extra_trees_ovr,
    gradient_boosting_ovr,
    ada_boost,
    extra_trees,
    gradient_boosting,
    rf_standard,
    rf_one_vs_rest,
    rf_one_vs_one
)

warnings.filterwarnings("ignore")

class Tee:
    """Escriu al terminal i al fitxer de log alhora."""
    def __init__(self, logfile_path):
        # stdout real
        self.terminal = sys.__stdout__
        # fitxer on desarem TOTA la sortida
        self.log = open(logfile_path, "w", encoding="utf-8")  # 'w' per sobreescriure cada execució

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # per compatibilitat amb wrappers que criden flush()
        self.terminal.flush()
        self.log.flush()
@contextlib.contextmanager
def suppress_stdout_stderr():
    """Un context manager que redirigeix stdout i stderr a /dev/null."""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


# Configurem Tee abans de cap print
log_dir = os.path.dirname(__file__)                   # directori MODELS/RandomForest
log_file = os.path.join(log_dir, "output.txt")        # MODELS/RandomForest/output.txt
sys.stdout = Tee(log_file)
sys.stderr = sys.stdout   # opcional: també errors al fitxer

# ============================
#  FUNCIONS AUXILIARS
# ============================

def print_report(model_name, vector_method, time_taken, y_true, y_pred, title):
    """Imprimeix un informe de classificació per a un split."""
    print(f"--- {title} ---")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(classification_report(y_true, y_pred, digits=4))
    print("-" * 60)


def main():
   

    # Diferents max_features que vols provar
    #max_features_list = [500, 5000, 50000, 80000]

    # Definim els models i els paràmetres
    # Cada entrada: (nom_model, funció_constructora, dict_paràmetres)
    models_to_test = [
        ("lightgbm_multiclass", lightgbm_multiclass, {"n_estimators": 200, "max_depth": 100}),
        ("lightgbm_ovr", lightgbm_ovr, {"n_estimators": 200, "max_depth": 100}),
        ##("extra_trees_ovr", extra_trees_ovr, {"n_estimators": 100}),
        ##("ada_boost_ovr", ada_boost_ovr, {"n_estimators": 100, "learning_rate": 0.5}),
        ##("gradient_boosting_ovr", gradient_boosting_ovr, {"n_estimators": 100, "learning_rate": 0.1}),
        ##("ada_boost", ada_boost, {"n_estimators": 100}),
        ###("extra_trees", extra_trees, {"n_estimators": 100}),
        ##("gradient_boosting", gradient_boosting, {"n_estimators": 100}),
     #   ("rf_standard", rf_standard, {"n_estimators": 200, "max_depth": 100}),
      #  ("rf_one_vs_rest", rf_one_vs_rest, {"n_estimators": 200, "max_depth": 100}),
       # ("rf_one_vs_one", rf_one_vs_one, {"n_estimators": 200, "max_depth": 100}),
    ]
    # Vectoritzacions a provar (pots afegir "BOW", "SVD", etc. si load_and_vectorize_splits ho suporta)
    vectorizers_to_test = ["TFIDF"]
    
    for vec_method in vectorizers_to_test:
        # Carreguem i vectoritzem splits
        data = load_and_vectorize_splits(method=vec_method)
        
        X_train = data["X_train"]
        X_val = data["X_val"]
        X_test = data["X_test"]
        y_train = data["y_train"]
        y_val = data["y_val"]
        y_test = data["y_test"]

        for model_name, builder_fn, params in models_to_test:
            print("\n------------------------------------------------------------")
            print(f"Model: {model_name} | Vectorització: {vec_method}")
            print("------------------------------------------------------------\n")
            
            start_time = time.time()
            with suppress_stdout_stderr():
                model = builder_fn(X_train, y_train, **params)
            end_time = time.time()

            time_taken = end_time - start_time
            print(f"Temps entrenament: {time_taken:.2f} s\n")

            # --- Avaluació ---
            # TRAIN
            train_pred = model.predict(X_train)
            print_report(model_name, vec_method, time_taken, y_train, train_pred, "TRAIN")

            # VALIDATION
            val_pred = model.predict(X_val)
            print_report(model_name, vec_method, time_taken, y_val, val_pred, "VALIDATION")

            # TEST
            test_pred = model.predict(X_test)
            print_report(model_name, vec_method, time_taken, y_test, test_pred, "TEST")

            # Si és un GridSearchCV, imprimeix millors paràmetres (si existeixen)
            if hasattr(model, "best_params_"):
                print("Best params (GridSearch):")
                print(model.best_params_)
                print("-" * 60)

    print(f"\nResults also saved to {log_file}")

if __name__ == "__main__":
    main()