import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
import time
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


from AnalizarLimpiarDividir.vector_representation import load_and_vectorize_splits
from MODELS.LogisticRegression.logistic_regression import (
    model_standard, model_one_vs_one, model_one_vs_rest, model_grid_search
)

# ============================
#  Tee per duplicar sortida a fitxer i terminal
# ============================

class Tee:
    """Escriu al terminal i a un fitxer alhora."""
    def __init__(self, logfile_path):
        self.terminal = sys.__stdout__      # stdout real
        self.log = open(logfile_path, "w", encoding="utf-8")  # 'w' per sobreescriure cada execució

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # Necessari perquè alguns wrappers de stdout ho criden
        self.terminal.flush()
        self.log.flush()

# Configurem el log abans de fer cap print
log_dir = os.path.dirname(__file__)                  # directori /MODELS/LogisticRegression/
log_file = os.path.join(log_dir, "output.txt")       # /MODELS/LogisticRegression/output.txt
sys.stdout = Tee(log_file)

def print_report(y_true, y_pred, title):
    """Prints a formatted report for a given dataset split."""
    print(f"--- {title} ---")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(classification_report(y_true, y_pred, digits=4))
    print("-" * 60)

def main():
    """
    Main function to run the logistic regression benchmark.
    """
    # --- Models and Vectorizers to Test ---
    models_to_test = [
        ("standard", model_standard),
        ("ovo", model_one_vs_one),
        ("ovr", model_one_vs_rest)
        #("grid", model_grid_search) # Excluded by default as it's very slow
    ]
    vectorizers_to_test = ['TFIDF', 'BOW']

    # --- Main Benchmark Loop ---
    for vec_method in vectorizers_to_test:
        
        data = load_and_vectorize_splits(method=vec_method)
        
        # Extreiem tot del diccionari
        X_train    = data["X_train"]
        X_val      = data["X_val"]
        X_test     = data["X_test"]
        y_train    = data["y_train"]
        y_val      = data["y_val"]
        y_test     = data["y_test"]
        

        for model_name, model_fn in models_to_test:
            print(f"\n============================================================")
            print(f"Model: {model_name} | Vectorització: {vec_method}")
            print(f"============================================================\n")
            
            start_time = time.time()
            model = model_fn(X_train, y_train)
            end_time = time.time()
            
            time_taken = end_time - start_time
            print(f"Temps entrenament: {time_taken:.2f} s\n")

            # --- Evaluation ---
            # Train set
            train_pred = model.predict(X_train)
            print_report(y_train, train_pred, "TRAIN")
            
            # Validation set
            val_pred = model.predict(X_val)
            print_report(y_val, val_pred, "VALIDATION")
            
            # Test set
            test_pred = model.predict(X_test)
            print_report(y_test, test_pred, "TEST")


if __name__ == "__main__":
    main()
