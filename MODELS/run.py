import pandas as pd
import time
from sklearn.metrics import classification_report, accuracy_score
import vector_representation as vr
import logistic_regression as lr

# ==========================================
# CONFIGURACIÓ GLOBAL
# ==========================================
SELECTED_MODEL = 'ovr'   # 'standard', 'ovo', 'ovr', 'grid'
OUTPUT_FILE = "output.txt"

# Hiperparàmetres per als models (Logistic Regression)
# Pots canviar C, max_iter, penalty, class_weight, etc.
HYPERPARAMETERS = {
    'C': 2.0,
    'max_iter': 500,
    'penalty': 'l2',
    'class_weight': 'balanced' 
}
# ==========================================

def save_report(file_handle, title, y_true, y_pred):
    file_handle.write(f"\n--- {title} ---\n")
    file_handle.write(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}\n")
    file_handle.write(classification_report(y_true, y_pred))
    file_handle.write("-" * 60 + "\n")

def main():
    start_time = time.time()

    # 1. Carregar dades
    print("Carregant datasets...")
    train = pd.read_csv("DATASETS/SPLIT/twitter_trainBALANCED.csv")
    val = pd.read_csv("DATASETS/SPLIT/twitter_valBALANCED.csv")
    test = pd.read_csv("DATASETS/SPLIT/twitter_testBALANCED.csv")

    # 2. Vectorització
    X_train, X_val, X_test, _ = vr.get_vectors(train['text'], val['text'], test['text'])
    
    y_train = train['label']
    y_val = val['label']
    y_test = test['label']

    # 3. Selecció i entrenament
    print(f"Entrenant model: {SELECTED_MODEL.upper()} amb params: {HYPERPARAMETERS}...")
    
    # Passem el diccionari HYPERPARAMETERS desempaquetat (**HYPERPARAMETERS)
    if SELECTED_MODEL == 'standard':
        model = lr.model_standard(X_train, y_train, **HYPERPARAMETERS)
    elif SELECTED_MODEL == 'ovo':
        model = lr.model_one_vs_one(X_train, y_train, **HYPERPARAMETERS)
    elif SELECTED_MODEL == 'ovr':
        model = lr.model_one_vs_rest(X_train, y_train, **HYPERPARAMETERS)
    elif SELECTED_MODEL == 'grid':
        # GridSearch defineix els seus propis params internament, ignorem els globals
        model = lr.model_grid_search(X_train, y_train)
    else:
        raise ValueError("Model no reconegut.")

    # 4. Prediccions
    print("Generant prediccions...")
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    # Càlcul del temps
    end_time = time.time()
    total_seconds = end_time - start_time
    
    if total_seconds >= 60:
        mins = int(total_seconds // 60)
        secs = total_seconds % 60
        time_str = f"{mins} min {secs:.2f} s"
    else:
        time_str = f"{total_seconds:.2f} s"

    # 5. Guardar resultats
    print(f"Guardant resultats a {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w") as f:
        f.write(f"RESUM EXECUCIÓ\n")
        f.write(f"Model: {SELECTED_MODEL}\n")
        f.write(f"Hiperparàmetres: {HYPERPARAMETERS}\n")
        f.write(f"Temps total d'execució: {time_str}\n")
        f.write("=" * 30 + "\n")
        
        save_report(f, "TRAIN", y_train, train_pred)
        save_report(f, "VALIDATION", y_val, val_pred)
        save_report(f, "TEST", y_test, test_pred)
        
    print(f"Fet en {time_str}.")

if __name__ == "__main__":
    main()