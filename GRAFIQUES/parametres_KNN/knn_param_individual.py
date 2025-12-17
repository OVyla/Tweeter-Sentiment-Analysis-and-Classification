import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import sys, os

import sys
import os

# --- INICI BLOC AUTO-CONFIGURACIÓ PATH ---
# Aquest codi puja nivells fins a trobar la carpeta 'AnalizarLimpiarDividir'
current_dir = os.path.dirname(os.path.abspath(__file__))
while current_dir != os.path.dirname(current_dir):  # Evita bucle infinit al root del sistema
    if os.path.exists(os.path.join(current_dir, 'AnalizarLimpiarDividir')):
        sys.path.append(os.path.join(current_dir, 'AnalizarLimpiarDividir'))
        break
    current_dir = os.path.dirname(current_dir)
# -----------------------------------------

import vector_representation as vr
project_root = current_dir
from sklearn.preprocessing import LabelEncoder

# Load vectorized data
vectors_path = os.path.join(project_root, 'DATASETS', 'VECTORS', 'tfidf')
X_train, X_val, X_test, _ = vr.load_tfidf(prefix=vectors_path)

datasets_dir = os.path.join(project_root, 'DATASETS', 'SPLIT')
train = pd.read_csv(os.path.join(datasets_dir, 'twitter_trainBALANCED.csv')).sample(frac=0.05, random_state=42)
val = pd.read_csv(os.path.join(datasets_dir, 'twitter_valBALANCED.csv')).sample(frac=0.05, random_state=42)
test = pd.read_csv(os.path.join(datasets_dir, 'twitter_testBALANCED.csv')).sample(frac=0.05, random_state=42)
y_train = train['label'].reset_index(drop=True)
y_val = val['label'].reset_index(drop=True)
y_test = test['label'].reset_index(drop=True)

X_train = X_train[train.index]
X_val = X_val[val.index]
X_test = X_test[test.index]

# Parámetros a probar
param_grid = {
    'n_neighbors': [0,1, 3, 5, 7, 15, 30, 50,80,150,200],
    'metric': ['cosine', 'euclidean', 'manhattan', 'minkowski'],
    'algorithm': ['auto', 'brute', 'ball_tree', 'kd_tree'],
}

results = []
for param, values in param_grid.items():
    print(f"\nProbando parámetro: {param}")
    total = len(values)
    for idx, v in enumerate(values):
        print(f"  [{idx+1}/{total}] Probando {param} = {v} ...")
        kwargs = {
            'n_neighbors': 7,
            'metric': 'cosine',
            'algorithm': 'auto',
            'n_jobs': -1
        }
        kwargs[param] = v
        # Saltar combinaciones no válidas
        if param == 'metric' and v == 'cosine' and kwargs.get('algorithm') in ['kd_tree', 'ball_tree']:
            print(f"Saltando combinación no válida: metric=cosine con algorithm={kwargs.get('algorithm')}")
            continue
        if param == 'algorithm' and v in ['kd_tree', 'ball_tree'] and kwargs.get('metric') == 'cosine':
            print(f"Saltando combinación no válida: algorithm={v} con metric=cosine")
            continue
        # Para métricas numéricas, convertir etiquetas a números
        use_label_encoder = False
        if param == 'metric' and v in ['euclidean', 'manhattan', 'minkowski']:
            use_label_encoder = True
        if param != 'metric' and kwargs.get('metric') in ['euclidean', 'manhattan', 'minkowski']:
            use_label_encoder = True
        if use_label_encoder:
            le = LabelEncoder()
            y_train_enc = le.fit_transform(y_train)
            y_test_enc = le.transform(y_test)
        else:
            y_train_enc = y_train
            y_test_enc = y_test
        try:
            knn = KNeighborsClassifier(**kwargs)
            knn.fit(X_train, y_train_enc)
            train_preds = knn.predict(X_train)
            test_preds = knn.predict(X_test)
            acc_train = accuracy_score(y_train_enc, train_preds)
            acc_test = accuracy_score(y_test_enc, test_preds)
            print(f"{param}={v}: Train={acc_train:.4f} | Test={acc_test:.4f}")
            results.append((param, v, acc_train, acc_test))
        except Exception as e:
            print(f"Error con {param}={v}: {e}")

# Guardar resultados
results_df = pd.DataFrame(results, columns=['param', 'value', 'train_accuracy', 'test_accuracy'])
results_df.to_csv(os.path.join(current_dir, 'knn_param_individual_results.csv'), index=False)


# Graficar resultados
import matplotlib.pyplot as plt
cat_params = ['metric', 'algorithm']
num_params = ['n_neighbors']

for param in cat_params:
    df = results_df[results_df['param'] == param]
    plt.figure(figsize=(8,5))
    x = df['value'].astype(str)
    plt.bar(x, df['train_accuracy'], color='lightgreen', alpha=0.7, label='Train')
    plt.bar(x, df['test_accuracy'], color='salmon', alpha=0.7, label='Test', bottom=df['train_accuracy']*0)
    plt.xlabel(param)
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy según {param} (KNN)')
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(current_dir, f'knn_bar_{param}_train_test.png'))
    plt.close()
    print(f'Guardada gráfica de barras para {param}: knn_bar_{param}_train_test.png')

for param in num_params:
    df = results_df[results_df['param'] == param]
    plt.figure(figsize=(8,5))
    plt.plot(df['value'], df['train_accuracy'], marker='o', color='blue', label='Train')
    plt.plot(df['value'], df['test_accuracy'], marker='^', color='orange', label='Test')
    plt.xlabel(param)
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy según {param} (KNN)')
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(current_dir, f'knn_line_{param}_train_test.png'))
    plt.close()
    print(f'Guardada gráfica de línea para {param}: knn_line_{param}_train_test.png')

print('Resultados guardados en knn_param_individual_results.csv')
