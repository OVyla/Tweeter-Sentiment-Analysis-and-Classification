from datasets import load_dataset
import pandas as pd

# ------------------------------
# 1️⃣ Cargar SOLO el train del dataset
# ------------------------------
print("Cargando bdstar/Tweets-Sentiment-Analysis (solo train)...")

# Cargar SOLO el split de train
train_ds = load_dataset("bdstar/Tweets-Sentiment-Analysis", split="train")

# Convertir a DataFrame - usando la columna 'label' que ya existe
full_dataset = pd.DataFrame({
    "text": train_ds['text'], 
    "label": train_ds['label']  # Ya viene con 'negative', 'neutral', 'positive'
})

print("Distribución original del dataset TRAIN:")
print(full_dataset['label'].value_counts())
print(f"Total de muestras: {len(full_dataset)}")

# ------------------------------
# 2️⃣ Encontrar la clase MENOS frecuente
# ------------------------------
label_counts = full_dataset['label'].value_counts()
min_label = label_counts.idxmin()
min_count = label_counts[min_label]

print(f"\nClase menos frecuente: '{min_label}' con {min_count} muestras")
print("Distribución completa:")
for label, count in label_counts.items():
    percentage = (count / len(full_dataset)) * 100
    print(f"  {label}: {count} muestras ({percentage:.1f}%)")

# ------------------------------
# 3️⃣ Balancear: tomar misma cantidad de cada clase
# ------------------------------
balanced_dfs = []

for label in full_dataset['label'].unique():
    df_label = full_dataset[full_dataset['label'] == label]
    
    # Si esta clase tiene más muestras que la mínima, hacer subsampling
    if len(df_label) > min_count:
        df_label = df_label.sample(n=min_count, random_state=42)
    
    balanced_dfs.append(df_label)
    print(f"  {label}: {len(df_label)} muestras (balanceadas)")

# Combinar todas las clases balanceadas
balanced_dataset = pd.concat(balanced_dfs, ignore_index=True)

# Mezclar aleatoriamente
balanced_dataset = balanced_dataset.sample(frac=1, random_state=42).reset_index(drop=True)

# ------------------------------
# 4️⃣ Mostrar distribución final
# ------------------------------
print("\n" + "="*50)
print("DISTRIBUCIÓN FINAL BALANCEADA:")
print("="*50)
final_counts = balanced_dataset['label'].value_counts()
print(final_counts)
print(f"\nTotal de muestras balanceadas: {len(balanced_dataset)}")

# ------------------------------
# 5️⃣ Guardar dataset balanceado
# ------------------------------
balanced_dataset.to_csv("twitter_balancedNOCLEAN.csv", index=False)
print(f"\nDataset balanceado guardado como: twitter_balancedNOCLEAN.csv")
