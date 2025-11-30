import pandas as pd
import time
from sklearn.metrics import classification_report, accuracy_score
import vector_representation as vr
import logistic_regression as lr

# Hiperparàmetres globals
HYPERPARAMETERS = {
    'C': 2.0,
    'max_iter': 500,
    'penalty': 'l2',
    'class_weight': 'balanced' 
}

def save_report(file_handle, title, y_true, y_pred):
    file_handle.write(f"\n--- {title} ---\n")
    file_handle.write(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}\n")
    file_handle.write(classification_report(y_true, y_pred))
    file_handle.write("-" * 60 + "\n")

def prepare_data(vector_method):
    """
    Carrega els CSVs i vectoritza. Retorna les matrius X i els vectors y.
    S'executa només UNA vegada per tipus de vectorització.
    """
    print(f"\n[DATA] Carregant i vectoritzant amb {vector_method}...")
    
    train = pd.read_csv("DATASETS/SPLIT/twitter_trainBALANCED.csv")
    val = pd.read_csv("DATASETS/SPLIT/twitter_valBALANCED.csv")
    test = pd.read_csv("DATASETS/SPLIT/twitter_testBALANCED.csv")

    y_train = train['label']
    y_val = val['label']
    y_test = test['label']

    X_train, X_val, X_test, _ = vr.get_vectors(
        train['text'], 
        val['text'], 
        test['text'], 
        method=vector_method
    )
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def train_and_report(data, model_name, vector_method, output_file, write_mode='a'):
    """
    Rep les dades JA vectoritzades i entrena el model.
    """
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = data
    
    start_time = time.time()
    print(f">>> ENTRENANT: Model={model_name} (Vector={vector_method})")

    # Selecció del model
    if model_name == 'standard':
        model = lr.model_standard(X_train, y_train, **HYPERPARAMETERS)
    elif model_name == 'ovo':
        model = lr.model_one_vs_one(X_train, y_train, **HYPERPARAMETERS)
    elif model_name == 'ovr':
        model = lr.model_one_vs_rest(X_train, y_train, **HYPERPARAMETERS)
    elif model_name == 'grid':
        model = lr.model_grid_search(X_train, y_train)
    else:
        raise ValueError(f"Model {model_name} no reconegut.")

    # Prediccions
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    time_str = f"{time.time() - start_time:.2f} s"

    # Guardar resultats
    with open(output_file, write_mode) as f:
        f.write(f"\n{'='*40}\n")
        f.write(f"RESULTATS EXECUCIÓ\n")
        f.write(f"Model: {model_name} | Vectorització: {vector_method}\n")
        f.write(f"Temps entrenament: {time_str}\n")
        f.write(f"{'='*40}\n")
        
        save_report(f, "TRAIN", y_train, train_pred)
        save_report(f, "VALIDATION", y_val, val_pred)
        save_report(f, "TEST", y_test, test_pred)
        
    print(f"   Fet en {time_str}. Guardat a {output_file}.")

if __name__ == "__main__":
    # Execució "standalone" per defecte (manté funcionalitat original)
    data = prepare_data('TFIDF')
    train_and_report(data, 'ovr', 'TFIDF', 'output.txt', 'w')