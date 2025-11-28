import pandas as pd
import numpy as np
import sys
import os

# --- Config ---
DATA_PATH = "DATASETS/twitter_balancedCLEAN.csv"
OUTPUT_FILE = "analisis_resultats.txt"  # Nom del fitxer de sortida

# --- Carregar dataset ---
# Ho fem abans d'obrir el fitxer de text per si falla la càrrega
try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    raise FileNotFoundError(f"No trobo {DATA_PATH}. Descarrega'l i posa'l allà.")

# Guardem la referència original de la pantalla (terminal)
original_stdout = sys.stdout 

print(f"Generant informe a {OUTPUT_FILE}...")

# Obrim el fitxer i redirigim tots els prints cap a dins
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    sys.stdout = f  # <--- MÀGIA: A partir d'aquí, 'print' escriu al fitxer

    # ==========================================
    # INICI DEL TEU CODI D'ANÀLISI
    # ==========================================

    print("=== INFORME D'ANÀLISI DEL DATASET ===\n")

    # --- 1) Número de mostres ---
    n_samples = len(df)
    print("Número de mostres:", n_samples)

    # --- 2) Classes presents ---
    unique_targets = sorted(df['label'].unique().tolist())
    print("Classes trobades (valors de 'label'):", unique_targets)

    # Normalització a minúscules (opcional però recomanat)
    df['label'] = df['label'].astype(str).str.lower().str.strip()

    n_classes = df['label'].nunique()
    print("Número de classes (etiquetes):", n_classes)
    print(df['label'].value_counts())

    # --- 3) Distribució per classe ---
    counts = df['label'].value_counts()
    percentages = (counts / n_samples * 100).round(2)
    print("\nDistribució de classes (recompte i %):")
    print(pd.concat([counts, percentages.rename('percent')], axis=1))

    # --- 4) Exemples de tweets per classe ---
    for cls in df['label'].unique():
        if pd.isna(cls):
            continue
        print(f"\nExemples (3) de tweets {str(cls).upper()}:")
        tweets = df[df['label'] == cls]['text'].dropna()
        # Control per si hi ha menys de 3 tweets
        sample_n = min(3, len(tweets))
        if sample_n > 0:
            for i, tweet in enumerate(tweets.sample(sample_n, random_state=42).tolist(), 1):
                print(f"  {i}. {tweet}")

    # --- 5) Tipus de columnes ---
    print("\nTipus de columnes:")
    print(df.dtypes)

    # --- 6) Comprovacions ràpides ---
    print("\nNaNs per columna:")
    print(df.isna().sum())

    # --- Longitud de text ---
    df['text_len'] = df['text'].astype(str).map(len)
    print("\nEstadístiques de la longitud dels tweets:")
    print(df['text_len'].describe())

    # --- Duplicats ---
    print("Duplicats totals:", df.duplicated(subset=['text']).sum())

    # Estadístiques de longitud per classe
    df_no_nan = df.dropna(subset=['label'])
    print("\nEstadístiques de longitud per classe:")
    print(df_no_nan.groupby('label')['text_len'].describe())

    # --- Duplicats per classe ---
    dup_by_class = df[df.duplicated(subset=['text'], keep=False)]['label'].value_counts()
    print("\nDuplicats per classe:")
    print(dup_by_class)

    # --- Primers 20 textos duplicats ---
    duplicated_texts = df[df.duplicated(subset=['text'], keep=False)].sort_values('text')
    print("\nPrimers 20 textos duplicats:")
    # Fem servir to_string() perquè pandas no talli la taula al fitxer
    print(duplicated_texts.head(20)[['text', 'label']].to_string())

    # ==========================================
    # FINAL DEL TEU CODI D'ANÀLISI
    # ==========================================

# --- RESTAURACIÓ ---
# Tornem a connectar el print a la pantalla
sys.stdout = original_stdout

print(f"Fet! Pots consultar els resultats a: {OUTPUT_FILE}")