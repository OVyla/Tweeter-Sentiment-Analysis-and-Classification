import pandas as pd
from clean_dataset import (
    lowercase_strip, remove_punctuation_space, replace_links, remove_mentions,
    remove_currency, normalize_laughs_en, normalize_repeated_chars, fix_abbr_en,
    remove_special_characters, lemmatize_text, remove_empty_texts, remove_duplicates
)

# --- CONFIGURACIÓN ---
EXTERNAL_CSV = "training.1600000.processed.noemoticon.csv"
TEXT_COL_IDX = 5
LABEL_COL_IDX = 0
OUTPUT_CSV = "external_clean_balanced.csv"

# --- Cargar y limpiar ---
df = pd.read_csv(EXTERNAL_CSV, header=None, encoding='latin-1')

# Extraer texto y etiquetas
text_series = df[TEXT_COL_IDX]
label_series = df[LABEL_COL_IDX]



# Solo clases binarias (0 y 4)
df2 = pd.DataFrame({
    'text': text_series,
    'label': label_series
})
df2 = df2[df2['label'].isin([0, 4])].reset_index(drop=True)
# Convertir etiquetas 0->negative, 4->positive
df2['label'] = df2['label'].map({0: 'negative', 4: 'positive'})

# Limpiar texto
print("Limpiando texto...")
df2['text'] = df2['text'].apply(lowercase_strip)
df2['text'] = df2['text'].apply(remove_punctuation_space)
df2['text'] = df2['text'].apply(replace_links)
df2['text'] = df2['text'].apply(remove_mentions)
df2['text'] = df2['text'].apply(remove_currency)
df2['text'] = df2['text'].apply(normalize_laughs_en)
df2['text'] = df2['text'].apply(normalize_repeated_chars)
df2['text'] = df2['text'].apply(fix_abbr_en)
df2['text'] = df2['text'].apply(remove_special_characters)
df2['text'] = df2['text'].apply(lemmatize_text)
df2 = remove_empty_texts(df2)
df2 = remove_duplicates(df2)

# --- Balancear clases ---
counts = df2['label'].value_counts()
print("Distribución de clases antes de balancear:")
print(counts)

min_count = counts.min()
if counts['negative'] != counts['positive']:
    # Balancear: tomar min_count de cada clase
    df_balanced = pd.concat([
        df2[df2['label'] == 'negative'].sample(n=min_count, random_state=42),
        df2[df2['label'] == 'positive'].sample(n=min_count, random_state=42)
    ]).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Clases balanceadas a {min_count} ejemplos cada una.")
else:
    df_balanced = df2
    print("Las clases ya están balanceadas.")

# Guardar CSV limpio y balanceado como text,label
df_balanced = df_balanced[['text', 'label']]
df_balanced.to_csv(OUTPUT_CSV, index=False)
print(f"Guardado en {OUTPUT_CSV}")
