import pandas as pd
import numpy as np

# --- Config ---
DATA_PATH = "DATASETMASDATOS/twitter_balancedCLEAN.csv"  # nuevo archivo multiclass

# --- Cargar dataset ---
try:
    df = pd.read_csv(DATA_PATH)
    
except FileNotFoundError:
    raise FileNotFoundError(f"No encuentro {DATA_PATH}. Descárgalo y colócalo ahí.")

# --- 1) Número de muestras ---
n_samples = len(df)
print("Número de muestras:", n_samples)


# --- 2) Clases presentes ---
unique_targets = sorted(df['label'].unique().tolist())
print("Clases encontradas (valores de 'label'):", unique_targets)


# Si los valores ya son strings, no hace falta mapear
# Si quieres mapear a minúsculas para homogeneidad:
df['label'] = df['label'].astype(str).str.lower().str.strip()

n_classes = df['label'].nunique()
print("Número de clases (etiquetas):", n_classes)
print(df['label'].value_counts())

# --- 3) Distribución por clase ---
counts = df['label'].value_counts()
percentages = (counts / n_samples * 100).round(2)
print("\nDistribución de clases (conteo y %):")
print(pd.concat([counts, percentages.rename('percent')], axis=1))

# --- 4) Ejemplos de tweets por clase ---
for cls in df['label'].unique():
    if pd.isna(cls):
        continue  # Skip NaN classes
    print(f"\nEjemplos (3) de tweets {str(cls).upper()}:")
    tweets = df[df['label'] == cls]['text'].dropna()
    sample_n = min(3, len(tweets))
    for i, tweet in enumerate(tweets.sample(sample_n, random_state=42).tolist(), 1):
        print(f"  {i}. {tweet}")

# --- 5) Tipos de columnas ---
print("\nTipos de columnas:")
print(df.dtypes)

# --- 6) Comprobaciones rápidas ---
print("\nNaNs por columna:")
print(df.isna().sum())

# --- Longitud de texto ---
df['text_len'] = df['text'].astype(str).map(len)
print("\nEstadísticas de la longitud de los tweets:")
print(df['text_len'].describe())

# --- Duplicados ---
print("Duplicados:", df.duplicated(subset=['text']).sum())

# Estadísticas de longitud por clase
df_no_nan = df.dropna(subset=['label'])
print(df_no_nan.groupby('label')['text_len'].describe())

# --- Duplicados por clase ---
dup_by_class = df[df.duplicated(subset=['text'], keep=False)]['label'].value_counts()
print("\nDuplicados por clase:")
print(dup_by_class)

# --- Primeros 20 textos duplicados ---
duplicated_texts = df[df.duplicated(subset=['text'], keep=False)].sort_values('text')
print("\nPrimeros 20 textos duplicados:")
print(duplicated_texts.head(20)[['text', 'label']].to_string())
