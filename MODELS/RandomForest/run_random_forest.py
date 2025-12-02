import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
import sys
import os
import time

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

    # --- Models and Vectorizers to Test ---
    models_to_test = [
        (
            "Random Forest OVR",
            lambda X_train, y_train: rf_one_vs_rest(
                X_train, y_train, n_estimators=100, max_depth=20, n_jobs=-1, random_state=42
            )
        ),
        (
            "AdaBoost OVR",
            lambda X_train, y_train: ada_boost_ovr(
                X_train, y_train, n_estimators=50, random_state=42
            )
        ),
        (
            "LightGBM Multiclass",
            lambda X_train, y_train: lightgbm_multiclass(
                X_train, y_train, n_estimators=500, random_state=42, n_jobs=-1
            )
        )
    ]
    # Using only TF-IDF as it's generally better for text classification with these models
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
            # Train set
            train_pred = model.predict(X_train)
            print_report(model_name, vec_method, time_taken, y_train, train_pred, "TRAIN")
            
            # Validation set
            val_pred = model.predict(X_val)
            print_report(model_name, vec_method, time_taken, y_val, val_pred, "VALIDATION")
            
            # Test set
            test_pred = model.predict(X_test)
            print_report(model_name, vec_method, time_taken, y_test, test_pred, "TEST")

if __name__ == "__main__":
    main()