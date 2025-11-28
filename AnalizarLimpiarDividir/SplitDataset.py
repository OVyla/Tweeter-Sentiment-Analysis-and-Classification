import pandas as pd
from sklearn.model_selection import train_test_split
import os 

# --- Cargar dataset limpio ---
# Ensure you run this script from the project root folder
df = pd.read_csv("DATASETS/twitter_balancedCLEAN.csv")

# --- División estratificada: Train / Validation / Test ---
# 80% train, 10% val, 10% test
train_df, temp_df = train_test_split(
    df, test_size=0.2, stratify=df['label'], random_state=42
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42
)

# --- Crear directorio si no existe ---
output_dir = "DATASETS/SPLIT"
os.makedirs(output_dir, exist_ok=True)

# --- Guardar conjuntos ---
print(f"Saving files to {output_dir}...")
train_df.to_csv(f"{output_dir}/twitter_trainBALANCED.csv", index=False)
val_df.to_csv(f"{output_dir}/twitter_valBALANCED.csv", index=False)
test_df.to_csv(f"{output_dir}/twitter_testBALANCED.csv", index=False)

# --- Mostrar proporciones para comprobar estratificación ---
def show_stats(name, d):
    print(f"--- {name} ---")
    print(f"Samples: {len(d)}")
    print(d['label'].value_counts(normalize=True).round(3))
    print("")

show_stats("Train", train_df)
show_stats("Validation", val_df)
show_stats("Test", test_df)