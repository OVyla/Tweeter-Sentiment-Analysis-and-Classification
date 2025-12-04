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
from SVM.svm_model import svm_one_vs_rest, svm_one_vs_one

def load_data(base_dir):
    print("Loading datasets for SVM...")
    train_df = pd.read_csv(os.path.join(base_dir, 'DATASETS', 'SPLIT', 'twitter_trainBALANCED.csv'))
    val_df = pd.read_csv(os.path.join(base_dir, 'DATASETS', 'SPLIT', 'twitter_valBALANCED.csv'))
    test_df = pd.read_csv(os.path.join(base_dir, 'DATASETS', 'SPLIT', 'twitter_testBALANCED.csv'))
    return train_df, val_df, test_df

def print_report(y_true, y_pred, title):
    return f"""--- {title} ---
Accuracy: {accuracy_score(y_true, y_pred):.4f}
{classification_report(y_true, y_pred, digits=4)}
{'-'*60}
"""

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    train_df, val_df, test_df = load_data(base_dir)
    y_train, y_val, y_test = train_df['label'], val_df['label'], test_df['label']

    vectorizers_to_test = ['TFIDF']
    output_file_path = os.path.join(os.path.dirname(__file__), "output_svm.txt")
    output_lines = []

    for vec_method in vectorizers_to_test:
        X_train, X_val, X_test, _ = get_vectors(train_df['text'], val_df['text'], test_df['text'], method=vec_method)

        models_to_test = [
            ("Linear SVM OvR", svm_one_vs_rest, "svm_ovr.joblib"),
            ("Linear SVM OvO", svm_one_vs_one, "svm_ovo.joblib")
        ]

        for model_name, model_fn, model_file in models_to_test:
            header = f"\n============================================================\nModel: {model_name} | Vectorización: {vec_method}\n============================================================\n"
            print(header)
            output_lines.append(header)

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

            output_lines.append(f"Tiempo de entrenamiento/carga: {time_taken:.2f} s\n\n")
            output_lines.append(print_report(y_train, model.predict(X_train), "TRAIN"))
            output_lines.append(print_report(y_val, model.predict(X_val), "VALIDATION"))
            output_lines.append(print_report(y_test, model.predict(X_test), "TEST"))

    with open(output_file_path, "w", encoding="utf-8") as f:
        f.writelines(output_lines)

    print(f"\nResults also saved to {output_file_path}")

if __name__ == "__main__":
    main()
