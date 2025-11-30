import pandas as pd
import time
from sklearn.metrics import classification_report, accuracy_score
import vector_representation as vr
import logistic_regression as lr

# Hiperparàmetres globals (per models estàndard)
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

def run_pipeline(selected_model, vector_method, output_file, write_mode='w'):
    """
    Executa el flux complet (ETL -> Train -> Predict -> Save)
    amb la configuració especificada.
    """
    start_time = time.time()
    print(f"\n>>> EXECUTANT PIPELINE: Model={selected_model}, Vector={vector_method}")

    # 1. Carregar dades
    # (Idealment això es carregaria fora per eficiència, però mantenim l'estructura simple)
    train = pd.read_csv("DATASETS/SPLIT/twitter_trainBALANCED.csv")
    val = pd.read_csv("DATASETS/SPLIT/twitter_valBALANCED.csv")
    test = pd.read_csv("DATASETS/SPLIT/twitter_testBALANCED.csv")

    y_train = train['label']
    y_val = val['label']
    y_test = test['label']

    # 2. Vectorització
    X_train, X_val, X_test, _ = vr.get_vectors(
        train['text'], 
        val['text'], 
        test['text'], 
        method=vector_method
    )
    
    # 3. Entrenament
    if selected_model == 'standard':
        model = lr.model_standard(X_train, y_train, **HYPERPARAMETERS)
    elif selected_model == 'ovo':
        model = lr.model_one_vs_one(X_train, y_train, **HYPERPARAMETERS)
    elif selected_model == 'ovr':
        model = lr.model_one_vs_rest(X_train, y_train, **HYPERPARAMETERS)
    elif selected_model == 'grid':
        model = lr.model_grid_search(X_train, y_train)
    else:
        raise ValueError(f"Model {selected_model} no reconegut.")

    # 4. Prediccions
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    # Temps
    total_seconds = time.time() - start_time
    time_str = f"{total_seconds:.2f} s"

    # 5. Guardar resultats
    with open(output_file, write_mode) as f:
        f.write(f"\n{'='*40}\n")
        f.write(f"RESULTATS EXECUCIÓ\n")
        f.write(f"Model: {selected_model} | Vectorització: {vector_method}\n")
        f.write(f"Temps: {time_str}\n")
        f.write(f"{'='*40}\n")
        
        save_report(f, "TRAIN", y_train, train_pred)
        save_report(f, "VALIDATION", y_val, val_pred)
        save_report(f, "TEST", y_test, test_pred)
        
    print(f"Resultats guardats a {output_file} ({time_str})")

if __name__ == "__main__":
    # Execució per defecte si es crida directament
    run_pipeline('ovr', 'TFIDF', 'output.txt', 'w')