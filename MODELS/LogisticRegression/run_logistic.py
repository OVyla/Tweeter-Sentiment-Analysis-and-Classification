import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
import sys
import os
import time

# --- Path Correction ---
# Add the parent directory (MODELS) to the path to find shared modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Module Imports ---
from vector_representation import get_vectors
from LogisticRegression.logistic_regression import model_standard, model_one_vs_one, model_one_vs_rest, model_grid_search

def load_data(base_dir):
    """Loads the CSV datasets."""
    print("Loading datasets...")
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
    Main function to run the logistic regression benchmark.
    """
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    train_df, val_df, test_df = load_data(base_dir)

    y_train, y_val, y_test = train_df['label'], val_df['label'], test_df['label']

    # --- Models and Vectorizers to Test ---
    models_to_test = [
        ("standard", model_standard),
        ("ovo", model_one_vs_one),
        ("ovr", model_one_vs_rest)
        #("grid", model_grid_search) # Excluded by default as it's very slow
    ]
    vectorizers_to_test = ['TFIDF']

    # --- Main Benchmark Loop ---
    for vec_method in vectorizers_to_test:
        X_train, X_val, X_test, _ = get_vectors(
            train_df['text'], val_df['text'], test_df['text'], method=vec_method
        )
        
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
            # Train
            train_pred = model.predict(X_train)
            output_lines.append(print_report(y_train, train_pred, "TRAIN"))

            # Validation
            val_pred = model.predict(X_val)
            output_lines.append(print_report(y_val, val_pred, "VALIDATION"))

            # Test
            test_pred = model.predict(X_test)
            output_lines.append(print_report(y_test, test_pred, "TEST"))


    # --- Save results to file ---
    with open(output_file_path, "w", encoding="utf-8") as f:
        for line in output_lines:
            f.write(line)

    print(f"\nResults also saved to {output_file_path}")
    

if __name__ == "__main__":
    main()
