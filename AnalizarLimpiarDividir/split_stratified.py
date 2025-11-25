import pandas as pd
from sklearn.model_selection import train_test_split

# --- Cargar dataset limpio ---
df = pd.read_csv("twitter_balancedCLEAN.csv")

# --- División estratificada: Train / Validation / Test ---
# 80% train, 10% val, 10% test (ajustable)
train_df, temp_df = train_test_split(
    df, test_size=0.2, stratify=df['label'], random_state=42
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42
)

# --- Guardar conjuntos ---
train_df.to_csv("twitter_trainBALANCED.csv", index=False)
val_df.to_csv("twitter_valBALANCED.csv", index=False)
test_df.to_csv("twitter_testBALANCED.csv", index=False)

# --- Mostrar proporciones para comprobar estratificación ---
def show_stats(name, d):
    print(f"{name}: {len(d)} muestras")
    print(d['label'].value_counts(normalize=True).round(3))

show_stats("Train", train_df)
show_stats("Validation", val_df)
show_stats("Test", test_df)
