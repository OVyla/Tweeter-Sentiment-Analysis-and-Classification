import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score
import sys, os

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
project_root = current_dir

# Cargar todo el dataset para tuning completo
datasets_dir = os.path.join(project_root, 'data', 'SPLIT')
train = pd.read_csv(os.path.join(datasets_dir, "twitter_trainBALANCED.csv"))
test = pd.read_csv(os.path.join(datasets_dir, "twitter_testBALANCED.csv"))

vectors_path = os.path.join(project_root, 'data', 'VECTORS', 'tfidf')
X_train_full, _, X_test_full, _ = vr.load_tfidf(prefix=vectors_path)
X_train = X_train_full[train.index]
X_test = X_test_full[test.index]
y_train = train['label']
y_test = test['label']

# Parámetros a probar
param_grids = {
    'alpha': [0,0.1, 0.4, 0.8, 1.0, 2.0, 5.0, 10, 20],
    'binarize': [0.0, 0.03,0.05, 0.1, 0.15,0.2, 0.3, 0.4, 0.5]
}
default_params = {'alpha': 1.0, 'fit_prior': True, 'binarize': 0.0}

for param, values in param_grids.items():
    train_acc = []
    test_acc = []
    failed = []
    for v in values:
        params = default_params.copy()
        params[param] = v
        try:
            model = BernoulliNB(**params)
            model.fit(X_train, y_train)
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            train_acc.append(accuracy_score(y_train, train_pred))
            test_acc.append(accuracy_score(y_test, test_pred))
        except Exception as e:
            train_acc.append(np.nan)
            test_acc.append(np.nan)
            failed.append((v, str(e)))
    plot_values = [str(v) for v in values]
    plt.figure(figsize=(8,5))
    if param == 'alpha' or param == 'binarize':
        plt.plot(plot_values, train_acc, marker='o', label='Train Accuracy')
        plt.plot(plot_values, test_acc, marker='s', label='Test Accuracy')
        # Líneas horizontales para facilitar comparación
        for acc in train_acc + test_acc:
            plt.axhline(y=acc, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
    else:
        x = np.arange(len(plot_values))
        width = 0.35
        plt.bar(x - width/2, train_acc, width, label='Train Accuracy')
        plt.bar(x + width/2, test_acc, width, label='Test Accuracy')
        plt.xticks(x, plot_values)
    plt.xlabel(param)
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy de train y test según {param} (BernoulliNB)')
    plt.legend()
    plt.grid(True)
    # Ajuste del eje Y: cada cuadro avanza 0.02
    min_acc = min(train_acc + test_acc)
    max_acc = max(train_acc + test_acc)
    plt.ylim(min_acc - 0.03, max_acc + 0.03)
    plt.yticks(np.arange(round(min_acc - 0.03, 2), round(max_acc + 0.03, 2)+0.001, 0.02))
    plt.tight_layout()
    fname = os.path.join(current_dir, f'bernoulli_tuning_{param}.png')
    plt.savefig(fname)
    plt.close()
    print(f'Guardada gráfica {fname}')
    if failed:
        print(f"Valores fallidos para {param}: {failed}")
print('¡Tuning de BernoulliNB completado!')
