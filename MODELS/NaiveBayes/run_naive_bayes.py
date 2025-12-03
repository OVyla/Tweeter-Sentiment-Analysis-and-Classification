import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import sys
import os
import time

# --- Path Correction ---
# Add the parent directory (MODELS) to the path to find shared modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Module Imports ---
from vector_representation import get_vectors
from naive_bayes import model_grid_search

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

def main():
    """
    Main function to run the Naive Bayes benchmark.
    """
    # Determine the project's base directory to construct absolute paths
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    # Load the datasets
    train_df, val_df, test_df = load_data(base_dir)

    # Extract labels
    y_train, y_val, y_test = train_df['label'], val_df['label'], test_df['label']

    # --- Vectorization ---
    # Generate TF-IDF vectors from the text data
    print("Generating TF-IDF vectors...")
    X_train, X_val, X_test, _ = get_vectors(
        train_df['text'], val_df['text'], test_df['text'], method='TFIDF'
    )
    
    # --- Model Training and Evaluation ---
    print("\n============================================================")
    print(f"Model: Naive Bayes (Bernoulli) with GridSearchCV")
    print(f"============================================================\n")
    
    start_time = time.time()
    # Find the best model using Grid Search on the training data
    # Note: BernoulliNB is often used with binary features (like BoW presence), but we test it here.
    best_model = model_grid_search(X_train, y_train, model_type='bernoulli')
    end_time = time.time()
    
    time_taken = end_time - start_time
    print(f"Training and Grid Search Time: {time_taken:.2f} s\n")

    # --- Evaluation ---
    # Predict on all datasets
    train_preds = best_model.predict(X_train)
    val_preds = best_model.predict(X_val)
    test_preds = best_model.predict(X_test)

    # Calculate accuracy scores
    acc_train = accuracy_score(y_train, train_preds)
    acc_val = accuracy_score(y_val, val_preds)
    acc_test = accuracy_score(y_test, test_preds)
    
    # Generate confusion matrix and classification report for the test set
    cm_test = confusion_matrix(y_test, test_preds, labels=["negative", "neutral", "positive"])
    report_test = classification_report(y_test, test_preds, digits=3)

    # --- Output Results ---
    output_lines = []
    output_lines.append(f"=== NAIVE BAYES (Bernoulli with GridSearch) ===\n")
    output_lines.append(f"Best Parameters Found: {best_model.get_params()}\n\n")
    output_lines.append(f"Train Accuracy: {acc_train:.4f}\n")
    output_lines.append(f"Validation Accuracy: {acc_val:.4f}\n")
    output_lines.append(f"Test Accuracy: {acc_test:.4f}\n\n")
    output_lines.append("Confusion Matrix (Test):\n")
    output_lines.append(str(cm_test) + "\n")
    output_lines.append("Classification Report (Test):\n")
    output_lines.append(report_test + "\n")

    # Print results to the console
    for line in output_lines:
        print(line, end="")

    # Save results to an output file in the current directory
    output_file_path = os.path.join(os.path.dirname(__file__), "output.txt")
    with open(output_file_path, "w", encoding="utf-8") as f:
        for line in output_lines:
            f.write(line)
    
    print(f"\nResults also saved to {output_file_path}")

if __name__ == "__main__":
    main()
