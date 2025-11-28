
"""
Análisis exploratorio de un dataset de tweets multiclass.
Muestra estadísticas, distribución de clases, ejemplos y duplicados.
"""

import pandas as pd
import numpy as np

DATA_PATH = "twitter_balancedCLEAN.csv"  # nuevo archivo multiclass

def cargar_dataset(path):
    """Carga el dataset y lanza error si no existe."""
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"No encuentro {path}. Descárgalo y colócalo ahí.")

def mostrar_info_basica(df):
    """Imprime número de muestras y clases presentes."""
    n_samples = len(df)
    print(f"Número de muestras: {n_samples}")
    unique_targets = sorted(df['label'].unique().tolist())
    print(f"Clases encontradas (valores de 'label'): {unique_targets}")

def homogeneizar_labels(df):
    """Convierte las etiquetas a minúsculas y sin espacios."""
    df['label'] = df['label'].astype(str).str.lower().str.strip()
    return df

def mostrar_distribucion_clases(df):
    """Imprime la distribución de clases (conteo y %)."""
    n_samples = len(df)
    counts = df['label'].value_counts()
    percentages = (counts / n_samples * 100).round(2)
    print("\nDistribución de clases (conteo y %):")
    print(pd.concat([counts, percentages.rename('percent')], axis=1))

def mostrar_ejemplos_por_clase(df, n=3):
    """Muestra n ejemplos de tweets por clase."""
    for cls in df['label'].unique():
        if pd.isna(cls):
            continue
        print(f"\nEjemplos ({n}) de tweets {str(cls).upper()}:")
        tweets = df[df['label'] == cls]['text'].dropna()
        sample_n = min(n, len(tweets))
        for i, tweet in enumerate(tweets.sample(sample_n, random_state=42).tolist(), 1):
            print(f"  {i}. {tweet}")

def mostrar_tipos_columnas(df):
    """Imprime los tipos de columnas."""
    print("\nTipos de columnas:")
    print(df.dtypes)

def mostrar_nans(df):
    """Imprime el número de NaNs por columna."""
    print("\nNaNs por columna:")
    print(df.isna().sum())

def analizar_longitud_texto(df):
    """Agrega columna de longitud de texto y muestra estadísticas."""
    df['text_len'] = df['text'].astype(str).map(len)
    print("\nEstadísticas de la longitud de los tweets:")
    print(df['text_len'].describe())
    return df

def mostrar_duplicados(df):
    """Imprime número de duplicados y ejemplos por clase."""
    print("Duplicados:", df.duplicated(subset=['text']).sum())
    df_no_nan = df.dropna(subset=['label'])
    print(df_no_nan.groupby('label')['text_len'].describe())
    dup_by_class = df[df.duplicated(subset=['text'], keep=False)]['label'].value_counts()
    print("\nDuplicados por clase:")
    print(dup_by_class)
    duplicated_texts = df[df.duplicated(subset=['text'], keep=False)].sort_values('text')
    print("\nPrimeros 20 textos duplicados:")
    print(duplicated_texts.head(20)[['text', 'label']].to_string())

def main():
    df = cargar_dataset(DATA_PATH)
    mostrar_info_basica(df)
    df = homogeneizar_labels(df)
    n_classes = df['label'].nunique()
    print(f"Número de clases (etiquetas): {n_classes}")
    print(df['label'].value_counts())
    mostrar_distribucion_clases(df)
    mostrar_ejemplos_por_clase(df, n=3)
    mostrar_tipos_columnas(df)
    mostrar_nans(df)
    df = analizar_longitud_texto(df)
    mostrar_duplicados(df)

if __name__ == "__main__":
    main()
