import os
import pandas as pd
import joblib
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURACIÓN ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..', '..'))
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'SPLIT')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'data', 'models_FINAL') # Carpeta para guardar modelos finales
os.makedirs(MODEL_DIR, exist_ok=True)

def load_data():
    print("Cargando datasets completos (Train + Val)...")
    train_path = os.path.join(DATA_PATH, 'twitter_trainBALANCED.csv')
    val_path = os.path.join(DATA_PATH, 'twitter_valBALANCED.csv')

    df_train = pd.read_csv(train_path).dropna(subset=['text', 'label'])
    df_val = pd.read_csv(val_path).dropna(subset=['text', 'label'])
    
    return df_train, df_val

def train_final_model():
    start_time = time.time()
    df_train, df_val = load_data()

    X_train = df_train['text']
    y_train = df_train['label']
    X_val = df_val['text']
    y_val = df_val['label']

    print(f"\nEntrenando Random Forest Final con {len(df_train)} muestras...")
    print("Parámetros: n_estimators=300, max_depth=None, min_samples_leaf=1")

    # --- PIPELINE FINAL CON MEJORES HIPERPARÁMETROS ---
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('rf', RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=42,
            verbose=1
        ))
    ])

    # Entrenar
    pipeline.fit(X_train, y_train)

    # --- EVALUACIÓN FINAL ---
    print("\nEvaluando en Set de Validación...")
    y_pred = pipeline.predict(X_val)
    
    report = classification_report(y_val, y_pred)
    print("\n--- REPORTE FINAL DE CLASIFICACIÓN ---")
    print(report)

    # Guardar reporte en texto
    with open(os.path.join(BASE_DIR, 'final_rf_report.txt'), 'w') as f:
        f.write(report)

    # --- MATRIZ DE CONFUSIÓN ---
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=pipeline.classes_, yticklabels=pipeline.classes_)
    plt.title('Confusion Matrix - Final Random Forest')
    plt.ylabel('Real')
    plt.xlabel('Predicho')
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'final_rf_confusion_matrix.png'))
    print("Matriz de confusión guardada.")

    # --- GUARDAR MODELO ---
    model_path = os.path.join(MODEL_DIR, 'random_forest_final.joblib')
    joblib.dump(pipeline, model_path)
    print(f"\n¡Modelo guardado exitosamente en: {model_path}!")

    minutes = (time.time() - start_time) / 60
    print(f"Tiempo total: {minutes:.2f} minutos")

if __name__ == "__main__":
    train_final_model()