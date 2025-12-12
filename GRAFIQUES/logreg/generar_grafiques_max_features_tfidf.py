import os
import re
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# Config de paths
# ============================================================
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Input file with the results
INPUT_FILE = os.path.join(ROOT_DIR, "MODELS", "LogisticRegression", "output_features_tfidf.txt")

# Output folder
OUTPUT_DIR = os.path.join(ROOT_DIR, "GRAFIQUES", "logreg")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def parse_results(filepath):
    """
    Parses output_features_tfidf.txt and extracts:
      - max_features
      - TRAIN Accuracy
      - VALIDATION Accuracy
    """
    max_features_list = []
    train_acc_list = []
    val_acc_list = []

    current_max_features = None
    current_train_acc = None
    current_section = None  # 'TRAIN', 'VALIDATION', 'TEST' or None

    if not os.path.exists(filepath):
        print(f"ERROR: File not found at {filepath}")
        return [], [], []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # 1) Detect "Max Features: X" (Start of a new block)
            if "Max Features:" in line:
                # Reset temp variables for the new block
                current_train_acc = None 
                
                match = re.search(r"Max Features:\s*([0-9]+)", line)
                if match:
                    current_max_features = int(match.group(1))
                continue

            # 2) Detect Sections
            if line.startswith("--- TRAIN ---"):
                current_section = "TRAIN"
                continue
            if line.startswith("--- VALIDATION ---"):
                current_section = "VALIDATION"
                continue
            
            # 3) Capture TRAIN Accuracy
            if current_section == "TRAIN" and line.startswith("Accuracy:"):
                match = re.search(r"Accuracy:\s*([0-9.]+)", line)
                if match:
                    current_train_acc = float(match.group(1))
                continue

            # 4) Capture VALIDATION Accuracy & Save Record
            # We assume Validation comes AFTER Train in the log file
            if current_section == "VALIDATION" and line.startswith("Accuracy:"):
                match = re.search(r"Accuracy:\s*([0-9.]+)", line)
                if match:
                    val_acc = float(match.group(1))

                    # Only append if we have all necessary data
                    if current_max_features is not None and current_train_acc is not None:
                        max_features_list.append(current_max_features)
                        train_acc_list.append(current_train_acc)
                        val_acc_list.append(val_acc)
                continue

    # Sort by max_features to ensure the line plot is drawn correctly
    if not max_features_list:
        print("Warning: No data found. Check if the text file format matches the parser.")
        return [], [], []

    zipped = list(zip(max_features_list, train_acc_list, val_acc_list))
    zipped.sort(key=lambda x: x[0])

    return zip(*zipped)


def plot_accuracy_curves(max_features, train_acc, val_acc, output_dir):
    """
    Draws the standard Blue (Train) vs Orange (Val) plot.
    """
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")

    # Plot Lines
    plt.plot(max_features, train_acc, 'o-', label='Train Accuracy', color='#1f77b4', linewidth=2)
    plt.plot(max_features, val_acc, 's-', label='Validation Accuracy', color='#ff7f0e', linewidth=2)

    # Log scale is usually better for max_features (500, 1000, 5000...)
    plt.xscale('log') 
    
    plt.xlabel("Number of Features (max_features) [Log Scale]", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title("Logistic Regression Performance vs. Vocabulary Size", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)

    # Save
    out_path = os.path.join(output_dir, "logreg_tfidf_accuracy_vs_max_features.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"Plot saved successfully: {out_path}")


def main():
    max_features, train_acc, val_acc = parse_results(INPUT_FILE)

    if max_features:
        print(f"Found {len(max_features)} data points.")
        print("Max Features:", list(max_features))
        print("Train Acc:   ", list(train_acc))
        print("Val Acc:     ", list(val_acc))
        
        plot_accuracy_curves(max_features, train_acc, val_acc, OUTPUT_DIR)
    else:
        print("No data extracted. Please check the content of 'output_features_tfidf.txt'.")

if __name__ == "__main__":
    main()