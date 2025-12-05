import os
import sys
import time
from joblib import dump, load
from sklearn.metrics import classification_report, accuracy_score

# --- Path Correction ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# --- Module Imports ---
from AnalizarLimpiarDividir.vector_representation import load_and_vectorize_splits
from MODELS.RandomForest.random_forest_model import (
    rf_one_vs_rest,
    ada_boost_ovr,
    lightgbm_multiclass
)

def print_report(y_true, y_pred, title):
    """Genera i imprimeix un report formatat, i el retorna com a string."""
    report_str = (
        f"--- {title} ---\n"
        f"Accuracy: {accuracy_score(y_true, y_pred):.4f}\n"
        f"{classification_report(y_true, y_pred, digits=4)}"
        f"{'-'*60}\n"
    )
    print(report_str)
    return report_str

def main():
    """
    Main function to run the Random Forest & Ensemble benchmark.
    Ara fem servir load_and_vectorize_splits en lloc de llegir CSV + get_vectors.
    """

    # --- Vectorization ---
    # Fem servir només TF-IDF
    vectorizers_to_test = ['TFIDF']
    output_file_path = os.path.join(os.path.dirname(__file__), "output_randomforest.txt")
    output_lines = []

    # --- Main Benchmark Loop ---
    for vec_method in vectorizers_to_test:
        # Carreguem i vectoritzem splits amb la funció centralitzada
        data = load_and_vectorize_splits(method=vec_method)

        X_train = data["X_train"]
        X_val   = data["X_val"]
        X_test  = data["X_test"]
        y_train = data["y_train"]
        y_val   = data["y_val"]
        y_test  = data["y_test"]
        # vectorizer = data["vectorizer"]  # per si el vols guardar

        # --- Models a provar ---
        models_to_test = [
            (  
                "Random Forest OvR",
                lambda X, y: rf_one_vs_rest(X, y, n_estimators=100, max_depth=20, n_jobs=-1, random_state=42),
                "rf_ovr.joblib"
            ),
            (
                "AdaBoost OvR",
                lambda X, y: ada_boost_ovr(X, y, n_estimators=50, random_state=42),
                "ada_ovr.joblib"
            ),
            (
                "LightGBM Multiclass",
                lambda X, y: lightgbm_multiclass(X, y, n_estimators=500, random_state=42, n_jobs=-1),
                "lgbm_multiclass.joblib"
            ),
        ]

        for model_name, model_fn, model_file in models_to_test:
            header = (
                "\n============================================================\n"
                f"Model: {model_name} | Vectorització: {vec_method}\n"
                "============================================================\n"
            )
            print(header)
            output_lines.append(header)

            model_path = os.path.join(os.path.dirname(__file__), model_file)

            # --- Entrenar o carregar model ---
            if os.path.exists(model_path):
                msg = f"Cargando modelo guardado: {model_file}\n"
                print(msg)
                output_lines.append(msg)
                model = load(model_path)
                time_taken = 0.0
            else:
                msg = f"Entrenando modelo: {model_name}\n"
                print(msg)
                output_lines.append(msg)

                start_time = time.time()
                model = model_fn(X_train, y_train)
                end_time = time.time()
                time_taken = end_time - start_time

                dump(model, model_path)
                msg2 = f"Modelo guardado en {model_file}\n"
                print(msg2)
                output_lines.append(msg2)

            msg_time = f"Tiempo de entrenamiento/carga: {time_taken:.2f} s\n\n"
            print(msg_time)
            output_lines.append(msg_time)

            # --- Evaluación ---
            train_pred = model.predict(X_train)
            val_pred   = model.predict(X_val)
            test_pred  = model.predict(X_test)

            output_lines.append(print_report(y_train, train_pred, "TRAIN"))
            output_lines.append(print_report(y_val,   val_pred,   "VALIDATION"))
            output_lines.append(print_report(y_test,  test_pred,  "TEST"))

    # --- Guardar resultados ---
    with open(output_file_path, "w", encoding="utf-8") as f:
        for line in output_lines:
            f.write(line)

    print(f"\nResults also saved to {output_file_path}")

if __name__ == "__main__":
    main()