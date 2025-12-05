import os
import sys
import time
from sklearn.metrics import classification_report, accuracy_score
fromjoblib import dump, load

# ============================
#  PATHS DEL PROJECTE
# ============================

# Aquest fitxer està (probablement) a MODELS/SVM/run_svm.py
# Afegim el root del projecte (dos nivells amunt) al sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# --- Imports del projecte ---
from AnalizarLimpiarDividir.vector_representation import load_and_vectorize_splits
from MODELS.SVM.svm_model import svm_standard, svm_one_vs_rest, svm_one_vs_one

# ============================
#  CLASSE TEE: DUPLICAR SORTIDA
# ============================

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

# Configurem Tee abans de cap print
log_dir = os.path.dirname(__file__)              # directori on és run_svm.py
log_file = os.path.join(log_dir, "output.txt")   # output.txt al mateix directori
sys.stdout = Tee(log_file)
sys.stderr = sys.stdout   # opcional: també errors cap al fitxer

# ============================
#  FUNCIONS AUXILIARS
# ============================

def print_report(model_name, vector_method, time_taken, y_true, y_pred, title):
    """Imprimeix un informe de classificació per a un split."""
    print(f"--- {title} ---")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(classification_report(y_true, y_pred, digits=4))
    print("-" * 60)

# ============================
#  MAIN
# ============================

def main():
    """
    Main function to run the SVM benchmark.
    Executa: Linear SVM OvR i Linear SVM OvO (definits a svm_model).
    """

    # --- Models a provar ---
    models_to_test = [
        ("svm_standard", svm_standard),
        ("svm_ovr", svm_one_vs_rest),
        ("svm_ovo", svm_one_vs_one),
    ]

    # --- Vectoritzadors a provar ---
    vectorizers_to_test = ['TFIDF']

    for vec_method in vectorizers_to_test:
        # Carregar i vectoritzar els splits amb el mètode indicat
        data = load_and_vectorize_splits(method=vec_method, max_features=60000)

        # Extreure dades del diccionari
        X_train = data["X_train"]
        X_val   = data["X_val"]
        X_test  = data["X_test"]
        y_train = data["y_train"]
        y_val   = data["y_val"]
        y_test  = data["y_test"]
        # vectorizer = data["vectorizer"]  # per si algun dia el vols guardar

        for model_name, model_fn in models_to_test:
            print("\n------------------------------------------------------------")
            print(f"Model: {model_name} | Vectorització: {vec_method} | vector_features=60000")
            print("------------------------------------------------------------\n")

            model_path = os.path.join(os.path.dirname(__file__), model_file)

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

    print(f"\nResults also saved to {log_file}")


if __name__ == "__main__":
    main()
