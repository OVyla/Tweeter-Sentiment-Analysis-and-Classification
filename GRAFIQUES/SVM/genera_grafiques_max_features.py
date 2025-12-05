import os
import re
import matplotlib.pyplot as plt

# ============================================================
# Config de paths
# ============================================================
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Fitxer d'entrada amb els resultats
INPUT_FILE = os.path.join( ROOT_DIR, "MODELS", "SVM", "output_features_tfidf.txt")

# Carpeta de sortida per les figures
OUTPUT_DIR = os.path.join(ROOT_DIR, "GRAFIQUES", "SVM")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def parse_results(filepath):
    """
    Llegeix el fitxer output_max_features_tfidf.txt i extreu:
      - max_features
      - temps d'entrenament
      - accuracy de VALIDATION

    Torna tres llistes: max_features_list, val_acc_list, train_time_list
    """
    max_features_list = []
    val_acc_list = []
    train_time_list = []

    current_max_features = None
    current_train_time = None
    current_section = None  # 'TRAIN', 'VALIDATION', 'TEST' o None

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # 1) Detectar línia amb "Max Features: X"
            if "Max Features:" in line:
                # Exemple: "Model: standard | Vectorització: TFIDF | Max Features: 500"
                match = re.search(r"Max Features:\s*([0-9]+)", line)
                if match:
                    current_max_features = int(match.group(1))
                continue

            # 2) Detectar "Temps entrenament: XX.XX s"
            if line.startswith("Temps entrenament:"):
                # Exemple: "Temps entrenament: 14.41 s"
                match = re.search(r"Temps entrenament:\s*([0-9.]+)", line)
                if match:
                    current_train_time = float(match.group(1))
                continue

            # 3) Detectar seccions TRAIN / VALIDATION / TEST
            if line.startswith("--- TRAIN ---"):
                current_section = "TRAIN"
                continue
            if line.startswith("--- VALIDATION ---"):
                current_section = "VALIDATION"
                continue
            if line.startswith("--- TEST ---"):
                current_section = "TEST"
                continue

            # 4) Quan estem a VALIDATION i trobem "Accuracy:"
            if current_section == "VALIDATION" and line.startswith("Accuracy:"):
                # Exemple: "Accuracy: 0.6979"
                match = re.search(r"Accuracy:\s*([0-9.]+)", line)
                if match and current_max_features is not None and current_train_time is not None:
                    val_acc = float(match.group(1))

                    max_features_list.append(current_max_features)
                    val_acc_list.append(val_acc)
                    train_time_list.append(current_train_time)

                    # Després de guardar, podem continuar.
                    # (No reiniciem res perquè el següent bloc tornarà
                    #  a sobreescriure current_max_features/temps.)
                continue

    # Ordenem per max_features per si l'ordre al fitxer no és estrictament creixent
    zipped = list(zip(max_features_list, val_acc_list, train_time_list))
    zipped.sort(key=lambda x: x[0])

    max_features_list = [z[0] for z in zipped]
    val_acc_list = [z[1] for z in zipped]
    train_time_list = [z[2] for z in zipped]

    return max_features_list, val_acc_list, train_time_list


def plot_validation_accuracy(max_features, val_acc, output_dir):
    """
    Dibuixa la recta Accuracy (VALIDATION) vs max_features.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(max_features, val_acc, marker='o')
    plt.xscale('log')  # opcional: com que els valors creixen molt, log va bé
    plt.xlabel("Nombre de característiques (max_features)")
    plt.ylabel("Accuracy (VALIDATION)")
    plt.title("SVM TF-IDF: Accuracy de validació vs max_features")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)

    out_path = os.path.join(output_dir, "svm_tfidf_val_accuracy_vs_max_features.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"Figura guardada: {out_path}")


def plot_training_time(max_features, train_time, output_dir):
    """
    Dibuixa la recta Temps d'entrenament vs max_features.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(max_features, train_time, marker='o')
    plt.xscale('log')  # opcional
    plt.xlabel("Nombre de característiques (max_features)")
    plt.ylabel("Temps d'entrenament (s)")
    plt.title("SVM TF-IDF: Temps d'entrenament vs max_features")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)

    out_path = os.path.join(output_dir, "svm_tfidf_train_time_vs_max_features.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"Figura guardada: {out_path}")


def main():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"No s'ha trobat el fitxer d'entrada: {INPUT_FILE}")

    max_features_list, val_acc_list, train_time_list = parse_results(INPUT_FILE)

    print("Max features:", max_features_list)
    print("Validation accuracy:", val_acc_list)
    print("Training time (s):", train_time_list)

    plot_validation_accuracy(max_features_list, val_acc_list, OUTPUT_DIR)
    plot_training_time(max_features_list, train_time_list, OUTPUT_DIR)


if __name__ == "__main__":
    main()