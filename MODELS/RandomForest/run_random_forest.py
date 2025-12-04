import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
import sys
import os
import time
from joblib import dump, load


# --- Path Correction ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Module Imports ---
from vector_representation import get_vectors
from RandomForest.random_forest_model import rf_one_vs_rest, ada_boost_ovr, lightgbm_multiclass

def load_data(base_dir):
    """Loads the CSV datasets."""
    print("Loading datasets for Random Forest...")
    try:
        train_df = pd.read_csv(os.path.join(base_dir, 'DATASETS', 'SPLIT', 'twitter_trainBALANCED.csv'))
        val_df = pd.read_csv(os.path.join(base_dir, 'DATASETS', 'SPLIT', 'twitter_valBALANCED.csv'))
        test_df = pd.read_csv(os.path.join(base_dir, 'DATASETS', 'SPLIT', 'twitter_testBALANCED.csv'))
        return train_df, val_df, test_df
    except FileNotFoundError as e:
        print(f"Error loading data: {e}. Make sure you have run the data preparation scripts.")
        sys.exit(1)

def print_report(model_name, vector_method, time_taken, y_true, y_pred, title):
    """Prints a formatted report for a given dataset split."""
    print(f"--- {title} ---")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(classification_report(y_true, y_pred, digits=4))
    print("-" * 60)

def main():
    """
    Main function to run the Random Forest & Ensemble benchmark.
    """
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    train_df, val_df, test_df = load_data(base_dir)

    y_train, y_val, y_test = train_df['label'], val_df['label'], test_df['label']

    # --- Vectorization ---
    # Using only TF-IDF as it's generally better for text classification with these models
    vectorizers_to_test = ['TFIDF']
    output_file_path = os.path.join(os.path.dirname(__file__), "output_randomforest.txt")
    output_lines = []

    # --- Main Benchmark Loop ---
    for vec_method in vectorizers_to_test:
        X_train, X_val, X_test, _ = get_vectors(
            train_df['text'], val_df['text'], test_df['text'], method=vec_method
        )
        
        # --- Modelos a probar ---
        models_to_test = [
            ("Random Forest OvR", lambda X, y: rf_one_vs_rest(X, y, n_estimators=100, max_depth=20, n_jobs=-1, random_state=42), "rf_ovr.joblib"),
            ("AdaBoost OvR", lambda X, y: ada_boost_ovr(X, y, n_estimators=50, random_state=42), "ada_ovr.joblib"),
            ("LightGBM Multiclass", lambda X, y: lightgbm_multiclass(X, y, n_estimators=500, random_state=42, n_jobs=-1), "lgbm_multiclass.joblib")
        ]

        for model_name, model_fn, model_file in models_to_test:
            header = f"\n============================================================\n"
            header += f"Model: {model_name} | Vectorización: {vec_method}\n"
            header += f"============================================================\n"
            print(header)
            output_lines.append(header)

            model_path = os.path.join(os.path.dirname(__file__), model_file)

            # --- Entrenar o cargar modelo ---
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

            msg_time = f"Tiempo de entrenamiento/carga: {time_taken:.2f} s\n\n"
            print(msg_time)
            output_lines.append(msg_time)

            # --- Evaluación ---
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)
            test_pred = model.predict(X_test)

            output_lines.append(print_report(y_train, train_pred, "TRAIN"))
            output_lines.append(print_report(y_val, val_pred, "VALIDATION"))
            output_lines.append(print_report(y_test, test_pred, "TEST"))

    # --- Guardar resultados ---
    with open(output_file_path, "w", encoding="utf-8") as f:
        for line in output_lines:
            f.write(line)

    print(f"\nResults also saved to {output_file_path}")


if __name__ == "__main__":
    main()
