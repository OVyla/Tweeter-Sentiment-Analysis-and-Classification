import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Trobar l'arrel del projecte buscant la carpeta 'data' o 'models' cap amunt
current = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(current, 'data')) and current != os.path.dirname(current):
    current = os.path.dirname(current)
project_root = current

# CONFIGURACIÓ
INPUT_FILE = os.path.join(project_root, "benchmark.txt")
OUTPUT_DIR = os.path.join(project_root, "plots", "Benchmark")

def parse_benchmark_file(filepath):
    """
    Llegeix el fitxer de sortida i extreu dades estructurades.
    """
    data = []
    
    # Patrons Regex per capturar la informació
    # Captura: Model: [grup1] | Vectorització: [grup2]
    re_model_vec = re.compile(r"Model:\s+(.*?)\s+\|\s+Vectorització:\s+(.*)")
    # Captura: Temps entrenament: [grup1] s
    re_time = re.compile(r"Temps entrenament:\s+([\d\.]+)\s+s")
    # Captura: Accuracy: [grup1]
    re_accuracy = re.compile(r"Accuracy:\s+([\d\.]+)")
    # Captura la fila de macro avg per agafar el F1-score (3a columna numèrica)
    re_macro_f1 = re.compile(r"macro avg\s+[\d\.]+\s+[\d\.]+\s+([\d\.]+)")

    current_entry = {}
    section = None # TRAIN, VALIDATION, TEST

    if not os.path.exists(filepath):
        print(f"Error: No s'ha trobat el fitxer {filepath}")
        return pd.DataFrame()

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        # 1. Detectar inici de nou model
        m_model = re_model_vec.search(line)
        if m_model:
            # Si teníem un model previ guardat, l'afegim a la llista
            if current_entry and 'Accuracy' in current_entry:
                data.append(current_entry)
            
            # Iniciem nova entrada
            current_entry = {
                'Model': m_model.group(1).strip(),
                'Vectorització': m_model.group(2).strip(),
                'Temps (s)': 0,
                'Accuracy': 0,
                'F1-Macro': 0
            }
            section = None
            continue

        # 2. Capturar Temps
        m_time = re_time.search(line)
        if m_time and current_entry:
            current_entry['Temps (s)'] = float(m_time.group(1))

        # 3. Detectar secció (Ens interessa TEST per comparar)
        if "--- TEST ---" in line:
            section = "TEST"
        elif "--- TRAIN ---" in line or "--- VALIDATION ---" in line:
            section = "OTHER"

        # 4. Capturar Mètriques (Només de la secció TEST)
        if section == "TEST" and current_entry:
            m_acc = re_accuracy.search(line)
            if m_acc:
                current_entry['Accuracy'] = float(m_acc.group(1))
            
            m_f1 = re_macro_f1.search(line)
            if m_f1:
                current_entry['F1-Macro'] = float(m_f1.group(1))

    # Afegir l'última entrada
    if current_entry and 'Accuracy' in current_entry:
        data.append(current_entry)

    return pd.DataFrame(data)

def generate_plots(df):
    """Genera i guarda les gràfiques comparatives."""
    if df.empty:
        print("No hi ha dades per graficar.")
        return

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Estil visual
    sns.set_theme(style="whitegrid")
    
    # 1. Gràfica d'ACCURACY
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df, x='Model', y='Accuracy', hue='Vectorització', palette="viridis")
    plt.title('Comparativa Accuracy (Test Set)', fontsize=16)
    plt.ylim(0.7, 0.85) # Ajustem eix Y per veure millor les diferències
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/benchmark_accuracy.png")
    print(f"Generat: {OUTPUT_DIR}/benchmark_accuracy.png")

    # 2. Gràfica de F1-MACRO
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df, x='Model', y='F1-Macro', hue='Vectorització', palette="magma")
    plt.title('Comparativa F1-Score Macro (Test Set)', fontsize=16)
    plt.ylim(0.7, 0.85)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/benchmark_f1.png")
    print(f"Generat: {OUTPUT_DIR}/benchmark_f1.png")

    # 3. Gràfica de TEMPS
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df, x='Model', y='Temps (s)', hue='Vectorització', palette="rocket")
    plt.title("Temps d'Entrenament (Segons)", fontsize=16)
    plt.yscale("log") # Escala logarítmica perquè el Grid/BOW triga molt més
    plt.ylabel("Segons (Escala Logarítmica)")
    for container in ax.containers:
        ax.bar_label(container, fmt='%.0f', padding=3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/benchmark_time.png")
    print(f"Generat: {OUTPUT_DIR}/benchmark_time.png")

def main():
    print("Analitzant benchmark_output.txt...")
    df = parse_benchmark_file(INPUT_FILE)
    
    if not df.empty:
        print("\nDades extretes:")
        print(df.to_string(index=False))
        print("\nGenerant gràfiques...")
        generate_plots(df)
    else:
        print("No s'han pogut extreure dades correctament.")

if __name__ == "__main__":
    main()