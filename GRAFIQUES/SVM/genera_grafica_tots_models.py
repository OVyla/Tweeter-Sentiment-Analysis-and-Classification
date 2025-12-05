import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ============================
# CONFIGURACIÓ DE RUTES
# ============================
PATH_INPUT_FILE = os.path.join("MODELS", "SVM", "output.txt")
PATH_OUTPUT_DIR = os.path.join("GRAFIQUES", "svm")
PATH_OUTPUT_IMAGE = os.path.join(PATH_OUTPUT_DIR, "benchmark_accuracy_comparison.png")

# ============================
# 1. FUNCIÓ PER PARSEJAR EL FITXER OUTPUT.TXT
# ============================
def parse_output_file(filepath):
    data = []
    current_model = None
    current_vec = None
    current_split = None

    header_pattern = re.compile(r"Model:\s+(.+?)\s+\|\s+Vectorització:\s+(.+)")
    split_pattern = re.compile(r"---\s+(TRAIN|VALIDATION|TEST)\s+---")
    accuracy_pattern = re.compile(r"Accuracy:\s+(\d+\.\d+)")

    if not os.path.exists(filepath):
        print(f"❌ ERROR: No s'ha trobat el fitxer: {filepath}")
        return []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            
            header_match = header_pattern.search(line)
            if header_match:
                current_model = header_match.group(1).strip()
                current_vec = header_match.group(2).strip()
                current_split = None 
                continue

            split_match = split_pattern.search(line)
            if split_match:
                current_split = split_match.group(1).capitalize()
                continue

            acc_match = accuracy_pattern.search(line)
            if acc_match and current_model and current_vec and current_split:
                accuracy_value = float(acc_match.group(1))
                
                data.append({
                    "Model_Base": current_model,
                    "Vectorization": current_vec,
                    "Model_Full": f"{current_model}\n({current_vec})",
                    "Split": current_split,
                    "Accuracy": accuracy_value
                })
                current_split = None

    return data

# ============================
# 2. FUNCIÓ PER GENERAR LA GRÀFICA
# ============================
def plot_benchmark_results(df, output_path):
    sns.set_theme(style="whitegrid")
    
    # Passem a percentatge
    df["Accuracy"] = df["Accuracy"] * 100

    custom_palette = {"Train": "#A8DADC", "Validation": "#457B9D", "Test": "#1D3557"}

    # --- CANVI 1: Més amplada (16) perquè les barres "respirin" ---
    plt.figure(figsize=(16, 8))

    ax = sns.barplot(
        data=df,
        x="Model_Full",
        y="Accuracy",
        hue="Split",
        palette=custom_palette,
        edgecolor="black",
        linewidth=0.5,
        # gap=0.1 # Opcional en versions molt noves de seaborn per separar grups
    )

    ax.set_title("Comparativa Accuracy - SVM", fontsize=18, pad=20, fontweight='bold')
    ax.set_ylabel("Accuracy (%)", fontsize=14)
    ax.set_xlabel("Model i Vectorització", fontsize=14)
    
    # Rotar etiquetes eix X perquè es llegeixin bé
    plt.xticks(rotation=45)

    # Marge superior extra per les etiquetes
    ax.set_ylim(0, 118)
    
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0, title="Split")

    # --- CANVI 2: Font més petita (8.5) ---
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f%%', padding=3, fontsize=8.5, fontweight='bold')

    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    print(f"✅ Gràfica guardada correctament a: {output_path}")

# ============================
# MAIN
# ============================
if __name__ == "__main__":
    print(f"🔍 Llegint dades de: {PATH_INPUT_FILE}...")
    
    parsed_data = parse_output_file(PATH_INPUT_FILE)

    if parsed_data:
        df_results = pd.DataFrame(parsed_data)
        
        if not os.path.exists(PATH_OUTPUT_DIR):
            os.makedirs(PATH_OUTPUT_DIR)
        
        plot_benchmark_results(df_results, PATH_OUTPUT_IMAGE)
    else:
        print("⚠️ No s'han trobat dades. Revisa el fitxer output.txt.")