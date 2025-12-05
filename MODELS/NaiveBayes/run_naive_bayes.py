import os
import sys
import time
from sklearn.metrics import classification_report, accuracy_score
from joblib import dump, load

# ============================
#  PATHS DEL PROJECTE
# ============================

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from AnalizarLimpiarDividir.vector_representation import load_and_vectorize_splits
from MODELS.NaiveBayes.naive_bayes import (
    model_complement,
    model_multinomial,
    model_bernoulli,
    # model_gaussian,
    model_grid_search,
)

# ============================
#  CLASSE TEE: DUPLICAR SORTIDA
# ============================

class Tee:
    def __init__(self, logfile_path):
        self.terminal = sys.__stdout__
        self.log = open(logfile_path, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

log_dir = os.path.dirname(__file__)
log_file = os.path.join(log_dir, "output.txt")
sys.stdout = Tee(log_file)
sys.stderr = sys.stdout

# ============================
#  FUNCIONS AUXILIARS
# ============================

def print_report(y_true, y_pred, title):
    print(f"--- {title} ---")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(classification_report(y_true, y_pred, digits=4))
    print("-" * 60)

# ============================
#  MAIN
# ============================

def main():
    models_to_test = [
        ("complement",  model_complement,  "nb_complement.joblib"),
        ("multinomial", model_multinomial, "nb_multinomial.joblib"),
        ("bernoulli",   model_bernoulli,   "nb_bernoulli.joblib"),
        ("grid_search", model_grid_search, "nb_gridsearch.joblib"),
    ]

    vectorizers_to_test = ['TFIDF', 'BOW']

    for vec_method in vectorizers_to_test:
        data = load_and_vectorize_splits(method=vec_method)

        X_train = data["X_train"]
        X_val   = data["X_val"]
        X_test  = data["X_test"]
        y_train = data["y_train"]
        y_val   = data["y_val"]
        y_test  = data["y_test"]

        for model_name, model_fn, model_file in models_to_test:
            print("\n------------------------------------------------------------")
            print(f"Model: {model_name} | Vectorització: {vec_method}")
            print("------------------------------------------------------------\n")

            # Añadimos el vectorizador al nombre del archivo
            model_filename = f"{model_file.replace('.joblib','')}_{vec_method}.joblib"
            model_path = os.path.join(log_dir, model_filename)

            # 🔹 Entrenar o cargar
            if os.path.exists(model_path):
                print(f"Cargando modelo guardado: {model_file}")
                model = load(model_path)
                time_taken = 0.0
            else:
                print(f"Entrenando modelo: {model_name}")
                start_time = time.time()
                model = model_fn(X_train, y_train)
                end_time = time.time()
                time_taken = end_time - start_time
                dump(model, model_path)
                print(f"Modelo guardado en {model_file}")

            print(f"Temps entrenament/càrrega: {time_taken:.2f} s\n")

            # --- Avaluació ---
            train_pred = model.predict(X_train)
            print_report(y_train, train_pred, "TRAIN")

            val_pred = model.predict(X_val)
            print_report(y_val, val_pred, "VALIDATION")

            test_pred = model.predict(X_test)
            print_report(y_test, test_pred, "TEST")

            if hasattr(model, "best_params_"):
                print("Best params (GridSearch):")
                print(model.best_params_)
                print("-" * 60)

    print(f"\nResults also saved to {log_file}")

if __name__ == "__main__":
    main()
