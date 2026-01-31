import pandas as pd
import numpy as np
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score
import sys, os
import warnings
warnings.filterwarnings('ignore')

import sys
import os

# --- INICI BLOC AUTO-CONFIGURACIÓ PATH ---
# Aquest codi puja nivells fins a trobar la carpeta 'preprocessing'
current_dir = os.path.dirname(os.path.abspath(__file__))
while current_dir != os.path.dirname(current_dir):  # Evita bucle infinit al root del sistema
    if os.path.exists(os.path.join(current_dir, 'preprocessing')):
        sys.path.append(os.path.join(current_dir, 'preprocessing'))
        break
    current_dir = os.path.dirname(current_dir)
# -----------------------------------------

import vector_representation as vr

# ==========================================
# CONFIGURACIÓN
# ==========================================
OUTPUT_FILE = os.path.join(current_dir, "knn_random_search_results.txt")
CSV_RESULTS = os.path.join(current_dir, "knn_random_search_results.csv")
N_ITER = 25  # Número de combinaciones aleatorias (reducido - KNN es lento)
SAMPLE_SIZE = 0.03  # 3% del dataset para tuning (KNN muy costoso)
CV_FOLDS = 2  # 2-fold cross-validation (más rápido)
RANDOM_STATE = 42

# ==========================================

def main():
    print("\n" + "="*80)
    print("K-NEAREST NEIGHBORS - RANDOM SEARCH (TODOS LOS PARÁMETROS)")
    print("="*80)
    
    start_time = time.time()
    
    # 1. Cargar datasets
    print("\n[1/5] Cargando datasets...")
    # (Ya se carga internamente en load_and_vectorize_splits)
    
    # 2. Cargar vectores TF-IDF
    print("[2/5] Cargando vectores TF-IDF...")
    data = vr.load_and_vectorize_splits(method='TFIDF')
    X_train_full = data['X_train']
    X_val_full = data['X_val']
    X_test_full = data['X_test']
    y_train_full = data['y_train'] # Es un numpy array segons vr.py
    y_val_full = data['y_val']
    y_test_full = data['y_test']

    # Convertir a Series si el codi de baix espera .iloc
    # vr.load_and_vectorize_splits retorna .values (numpy array) per y_train
    # Però el codi de baix fa: y_train_full.iloc[indices]
    # Així que convertim a Series per compatibilitat o canviem el codi de baix.
    # Convertirem a Series per assegurar compatibilitat amb .iloc
    y_train_full = pd.Series(y_train_full)
    y_val_full = pd.Series(y_val_full)
    y_test_full = pd.Series(y_test_full)
    
    # 3. Muestrear para tuning
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
        'n_neighbors': [3, 5, 7, 9, 11, 15, 20],  # Valores clave (menos que antes)
        'weights': ['uniform', 'distance'],  # Tipo de pesos
        'algorithm': ['auto', 'brute'],  # Solo opciones rápidas
        'metric': ['cosine', 'euclidean'],  # Solo 2 métricas (cosine ideal TF-IDF)
        'leaf_size': [30, 50],  # Menos valores
    }
    
    base_knn = KNeighborsClassifier()
    
    rs = RandomizedSearchCV(
        base_knn,
        param_dist,
        n_iter=N_ITER,
        cv=CV_FOLDS,
        scoring='accuracy',
        n_jobs=1,  # Sequential para poder mostrar progreso
        random_state=RANDOM_STATE,
        verbose=2  # Progreso detallado de cada fit
    )
    
    # Crear wrapper para mostrar progreso
    print(f"\n  Iniciando búsqueda aleatoria...")
    search_start = time.time()
    rs.fit(X_train, y_train)
    search_time = time.time() - search_start
    print(f"\n  ✓ Búsqueda completada en {search_time/60:.2f} minutos")
    
    # 5. Entrenar modelo final con 40% del dataset (KNN es muy costoso)
    print("\n[5/5] Muestreando 40% del dataset para entrenamiento final...")
    sample_size_final = 0.4
    
    n_train_final = int(X_train_full.shape[0] * sample_size_final)
    train_final_idx = np.random.RandomState(42).choice(X_train_full.shape[0], n_train_final, replace=False)
    X_train_final = X_train_full[train_final_idx]
    y_train_final = y_train_full.iloc[train_final_idx].reset_index(drop=True)
    
    n_val_final = int(X_val_full.shape[0] * sample_size_final)
    val_final_idx = np.random.RandomState(42).choice(X_val_full.shape[0], n_val_final, replace=False)
    X_val_final = X_val_full[val_final_idx]
    y_val_final = y_val_full.iloc[val_final_idx].reset_index(drop=True)
    
    n_test_final = int(X_test_full.shape[0] * sample_size_final)
    test_final_idx = np.random.RandomState(42).choice(X_test_full.shape[0], n_test_final, replace=False)
    X_test_final = X_test_full[test_final_idx]
    y_test_final = y_test_full.iloc[test_final_idx].reset_index(drop=True)
    
    print(f"  Train: {n_train_final} / Val: {n_val_final} / Test: {n_test_final}")
    
    print("  Entrenando modelo...")
    final_model = KNeighborsClassifier(**rs.best_params_)
    final_model.fit(X_train_final, y_train_final)
    print("  ✓ Modelo entrenado exitosamente")
    
    # Predicciones sobre los mismos 40%
    print("\n[PREDICCIONES] Generando predicciones sobre el 40%...")
    print("  - Prediciendo en train set...")
    train_pred = final_model.predict(X_train_final)
    print("  ✓ Train set completado")
    
    print("  - Prediciendo en validation set...")
    val_pred = final_model.predict(X_val_final)
    print("  ✓ Validation set completado")
    
    print("  - Prediciendo en test set...")
    test_pred = final_model.predict(X_test_final)
    print("  ✓ Test set completado")
    
    train_acc = accuracy_score(y_train_final, train_pred)
    val_acc = accuracy_score(y_val_final, val_pred)
    test_acc = accuracy_score(y_test_final, test_pred)
    
    print(f"\n[RESULTADOS PRELIMINARES]")
    print(f"  Train Accuracy (40%): {train_acc:.4f}")
    print(f"  Validation Accuracy (40%): {val_acc:.4f}")
    print(f"  Test Accuracy (40%): {test_acc:.4f}")
    
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
        f.write("K-NEAREST NEIGHBORS - RANDOM SEARCH (TODOS LOS PARÁMETROS)\n")
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
        f.write("RESULTADOS EN 40% DE LOS data\n")
        f.write("-"*80 + "\n")
        f.write(f"Train Accuracy (40%): {train_acc:.4f}\n")
        f.write(f"Validation Accuracy (40%): {val_acc:.4f}\n")
        f.write(f"Test Accuracy (40%): {test_acc:.4f}\n")
        
        f.write("\n" + "-"*80 + "\n")
        f.write("TRAIN CLASSIFICATION REPORT (40%)\n")
        f.write("-"*80 + "\n")
        f.write(classification_report(y_train_final, train_pred))
        
        f.write("\n" + "-"*80 + "\n")
        f.write("VALIDATION CLASSIFICATION REPORT (40%)\n")
        f.write("-"*80 + "\n")
        f.write(classification_report(y_val_final, val_pred))
        
        f.write("\n" + "-"*80 + "\n")
        f.write("TEST CLASSIFICATION REPORT (40%)\n")
        f.write("-"*80 + "\n")
        f.write(classification_report(y_test_final, test_pred))
    
    # 2. CSV con todos los resultados
    results_df = pd.DataFrame(rs.cv_results_)
    results_df.to_csv(CSV_RESULTS, index=False)
    
    print(f"\n✓ Resultados guardados en:")
    print(f"  - {OUTPUT_FILE}")
    print(f"  - {CSV_RESULTS}")
    
    print("\n" + "="*80)
    print("RESUMEN FINAL")
    print("="*80)
    print(f"CV Score ({SAMPLE_SIZE*100:.0f}% dataset para tuning): {rs.best_score_:.4f}")
    print(f"Train Accuracy (40%): {train_acc:.4f}")
    print(f"Validation Accuracy (40%): {val_acc:.4f}")
    print(f"Test Accuracy (40%): {test_acc:.4f}")
    print(f"Tiempo total: {time_str}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
