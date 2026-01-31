import os
import re
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# CONFIGURATION
# ============================================================
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Input file
INPUT_FILE = os.path.join(ROOT_DIR, "models", "NaiveBayes", "output_features_tfidf.txt")

# Output folder
OUTPUT_DIR = os.path.join(ROOT_DIR, "plots", "NaiveBayes")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def parse_results(filepath):
    """
    Parses output_features_tfidf.txt and extracts:
      - max_features
      - TRAIN Accuracy
      - VALIDATION Accuracy
      - Training Time
    """
    max_features_list = []
    train_acc_list = []
    val_acc_list = []
    train_time_list = []

    current_max_features = None
    current_train_acc = None
    current_train_time = None
    current_section = None 

    if not os.path.exists(filepath):
        print(f"ERROR: File not found at {filepath}")
        return [], [], [], []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # 1) New Block: "Max Features"
            if "Max Features:" in line:
                # Reset temp vars
                current_train_acc = None
                
                match = re.search(r"Max Features:\s*([0-9]+)", line)
                if match:
                    current_max_features = int(match.group(1))
                continue

            # 2) Training Time
            if line.startswith("Temps entrenament:"):
                match = re.search(r"Temps entrenament:\s*([0-9.]+)", line)
                if match:
                    current_train_time = float(match.group(1))
                continue

            # 3) Detect Sections
            if line.startswith("--- TRAIN ---"):
                current_section = "TRAIN"
                continue
            if line.startswith("--- VALIDATION ---"):
                current_section = "VALIDATION"
                continue

            # 4) Capture TRAIN Accuracy
            if current_section == "TRAIN" and line.startswith("Accuracy:"):
                match = re.search(r"Accuracy:\s*([0-9.]+)", line)
                if match:
                    current_train_acc = float(match.group(1))
                continue

            # 5) Capture VALIDATION Accuracy & Save Record
            if current_section == "VALIDATION" and line.startswith("Accuracy:"):
                match = re.search(r"Accuracy:\s*([0-9.]+)", line)
                if match:
                    val_acc = float(match.group(1))

                    # We save only if we have all pieces of data
                    if current_max_features is not None and current_train_acc is not None:
                        max_features_list.append(current_max_features)
                        train_acc_list.append(current_train_acc)
                        val_acc_list.append(val_acc)
                        # Use 0.0 if time wasn't captured for some reason, to keep lists aligned
                        train_time_list.append(current_train_time if current_train_time else 0.0)
                continue

    # Sort
    if not max_features_list:
        print("Warning: No data found.")
        return [], [], [], []

    zipped = list(zip(max_features_list, train_acc_list, val_acc_list, train_time_list))
    zipped.sort(key=lambda x: x[0])

    return zip(*zipped)


def plot_accuracy_curves(max_features, train_acc, val_acc, output_dir):
    """
    Plots Train (Blue) and Validation (Orange) Accuracy.
    """
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")

    plt.plot(max_features, train_acc, 'o-', label='Train Accuracy', color='#1f77b4', linewidth=2)
    plt.plot(max_features, val_acc, 's-', label='Validation Accuracy', color='#ff7f0e', linewidth=2)

    plt.xscale('log')
    plt.xlabel("Number of Features (max_features) [Log Scale]", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title("Naive Bayes Performance vs. Vocabulary Size", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)

    out_path = os.path.join(output_dir, "nb_tfidf_accuracy_vs_max_features.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Accuracy Plot saved: {out_path}")


def plot_training_time(max_features, train_time, output_dir):
    """
    Plots Training Time.
    """
    plt.figure(figsize=(8, 5))
    sns.set_style("whitegrid")
    
    plt.plot(max_features, train_time, 'o-', color='#2ca02c', linewidth=2) # Green for time
    plt.xscale('log')
    plt.xlabel("Number of Features (max_features)", fontsize=11)
    plt.ylabel("Training Time (seconds)", fontsize=11)
    plt.title("Naive Bayes Training Time", fontsize=13)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)

    out_path = os.path.join(output_dir, "nb_tfidf_train_time_vs_max_features.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Time Plot saved: {out_path}")


def main():
    results = parse_results(INPUT_FILE)
    if not results or not results[0]: # Check if lists are empty
        print("No data extracted.")
        return

    max_features, train_acc, val_acc, train_time = results

    print(f"Found {len(max_features)} records.")
    print("Max Features:", list(max_features))
    
    plot_accuracy_curves(max_features, train_acc, val_acc, OUTPUT_DIR)
    plot_training_time(max_features, train_time, OUTPUT_DIR)


if __name__ == "__main__":
    main()