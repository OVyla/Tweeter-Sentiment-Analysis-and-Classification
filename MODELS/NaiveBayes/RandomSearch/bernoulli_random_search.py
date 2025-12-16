import pandas as pd
import numpy as np
import time
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score
import sys, os
import warnings
warnings.filterwarnings('ignore')

# Añadir MODELOS al path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../../'))

if project_root not in sys.path:
    sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'AnalizarLimpiarDividir'))

import vector_representation as vr

# ==========================================
# CONFIGURACIÓN
# ==========================================
OUTPUT_FILE = os.path.join(current_dir, "bernoulli_random_search_results.txt")
CSV_RESULTS = os.path.join(current_dir, "bernoulli_random_search_results.csv")
N_ITER = 50  # Número de combinaciones aleatorias (aumentado para exploración profunda)
SAMPLE_SIZE = 0.2  # 20% del dataset para tuning (mejor representación)
CV_FOLDS = 3  # 3-fold para mejor validación
RANDOM_STATE = 42

# ==========================================

def main():
    print("\n" + "="*80)
    print("BERNOULLI NAIVE BAYES - RANDOM SEARCH (TODOS LOS PARÁMETROS)")
    print("="*80)
    
    start_time = time.time()
    
    # 1. Cargar datasets
    print("\n[1/5] Cargando datasets...")
    datasets_dir = os.path.join(project_root, 'DATASETS', 'SPLIT')
    train = pd.read_csv(os.path.join(datasets_dir, "twitter_trainBALANCED.csv"))
    val = pd.read_csv(os.path.join(datasets_dir, "twitter_valBALANCED.csv"))
    test = pd.read_csv(os.path.join(datasets_dir, "twitter_testBALANCED.csv"))
    
    # 2. Cargar vectores TF-IDF
    print("[2/5] Cargando vectores TF-IDF...")
    vectors_path = os.path.join(project_root, 'DATASETS', 'VECTORS', 'tfidf')
    X_train_full, X_val_full, X_test_full, _ = vr.load_tfidf(prefix=vectors_path)
    y_train_full = train['label']
    y_val_full = val['label']
    y_test_full = test['label']
    
    # 3. Muestrear 10% para tuning
    print(f"[3/5] Muestreando {int(SAMPLE_SIZE*100)}% del dataset para tuning...")
    n_samples_total = X_train_full.shape[0]
    n_samples = int(n_samples_total * SAMPLE_SIZE)
    indices = np.random.RandomState(RANDOM_STATE).choice(n_samples_total, n_samples, replace=False)
    X_train = X_train_full[indices]
    y_train = y_train_full.iloc[indices].reset_index(drop=True)
    
    print(f"  - Dataset original: {n_samples_total} muestras")
    print(f"  - Dataset para tuning: {n_samples} muestras")
    
    # 4. Grid de parámetros para RandomizedSearchCV
    print("\n[4/5] Ejecutando RandomizedSearchCV ({} iteraciones, {}-fold CV)...".format(N_ITER, CV_FOLDS))
    
    param_dist = {
        'alpha': np.logspace(-5, 4, 30),  # Rango más amplio [0.00001 a 10000]
        'binarize': np.linspace(0, 1.0, 25),  # Más valores de umbral
        'fit_prior': [True, False],  # Si calcular probabilidades previas
    }
    
    base_nb = BernoulliNB()
    
    rs = RandomizedSearchCV(
        base_nb,
        param_dist,
        n_iter=N_ITER,
        cv=CV_FOLDS,
        scoring='accuracy',
        n_jobs=1,
        random_state=RANDOM_STATE,
        verbose=2
    )
    
    rs.fit(X_train, y_train)
    
    # 5. Entrenar modelo final con 100% del dataset
    print("\n[5/5] Entrenando modelo final con 100% del dataset...")
    final_model = BernoulliNB(**rs.best_params_)
    final_model.fit(X_train_full, y_train_full)
    
    # Predicciones
    print("Generando predicciones...")
    train_pred = final_model.predict(X_train_full)
    val_pred = final_model.predict(X_val_full)
    test_pred = final_model.predict(X_test_full)
    
    train_acc = accuracy_score(y_train_full, train_pred)
    val_acc = accuracy_score(y_val_full, val_pred)
    test_acc = accuracy_score(y_test_full, test_pred)
    
    # Calcular tiempo
    end_time = time.time()
    total_seconds = end_time - start_time
    mins = int(total_seconds // 60)
    secs = total_seconds % 60
    time_str = f"{mins} min {secs:.2f} s"
    
    # Guardar resultados
    print("\nGuardando resultados...")
    
    # 1. Archivo de texto con resumen
    with open(OUTPUT_FILE, "w") as f:
        f.write("="*80 + "\n")
        f.write("BERNOULLI NAIVE BAYES - RANDOM SEARCH (TODOS LOS PARÁMETROS)\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Tiempo total de ejecución: {time_str}\n")
        f.write(f"Iteraciones de RandomizedSearchCV: {N_ITER}\n")
        f.write(f"K-fold cross-validation: {CV_FOLDS}\n")
        f.write(f"Dataset para tuning: {SAMPLE_SIZE*100:.0f}% ({X_train.shape[0]} muestras)\n")
        f.write(f"Dataset para evaluación: 100% ({X_train_full.shape[0]} muestras)\n")
        f.write("\n" + "-"*80 + "\n")
        f.write("MEJORES PARÁMETROS ENCONTRADOS\n")
        f.write("-"*80 + "\n")
        for param, value in sorted(rs.best_params_.items()):
            f.write(f"  {param}: {value}\n")
        f.write(f"\n  CV Score ({SAMPLE_SIZE*100:.0f}% dataset): {rs.best_score_:.4f}\n")
        
        f.write("\n" + "-"*80 + "\n")
        f.write("RESULTADOS EN DATASETS COMPLETOS (100%)\n")
        f.write("-"*80 + "\n")
        f.write(f"Train Accuracy: {train_acc:.4f}\n")
        f.write(f"Validation Accuracy: {val_acc:.4f}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        
        f.write("\n" + "-"*80 + "\n")
        f.write("TRAIN CLASSIFICATION REPORT\n")
        f.write("-"*80 + "\n")
        f.write(classification_report(y_train_full, train_pred))
        
        f.write("\n" + "-"*80 + "\n")
        f.write("VALIDATION CLASSIFICATION REPORT\n")
        f.write("-"*80 + "\n")
        f.write(classification_report(y_val_full, val_pred))
        
        f.write("\n" + "-"*80 + "\n")
        f.write("TEST CLASSIFICATION REPORT\n")
        f.write("-"*80 + "\n")
        f.write(classification_report(y_test_full, test_pred))
    
    # 2. CSV con todos los resultados
    results_df = pd.DataFrame(rs.cv_results_)
    results_df.to_csv(CSV_RESULTS, index=False)
    
    print(f"\n✓ Resultados guardados en:")
    print(f"  - {OUTPUT_FILE}")
    print(f"  - {CSV_RESULTS}")
    
    print("\n" + "="*80)
    print("RESUMEN FINAL")
    print("="*80)
    print(f"CV Score ({SAMPLE_SIZE*100:.0f}% dataset): {rs.best_score_:.4f}")
    print(f"Train Accuracy (100%): {train_acc:.4f}")
    print(f"Validation Accuracy (100%): {val_acc:.4f}")
    print(f"Test Accuracy (100%): {test_acc:.4f}")
    print(f"Tiempo total: {time_str}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
