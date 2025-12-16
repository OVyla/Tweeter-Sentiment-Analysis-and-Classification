import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import sys, os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))

if project_root not in sys.path:
    sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'AnalizarLimpiarDividir'))

import vector_representation as vr

# Cargar todo el dataset para tuning completo
datasets_dir = os.path.join(project_root, 'DATASETS', 'SPLIT')
train = pd.read_csv(os.path.join(datasets_dir, "twitter_trainBALANCED.csv"))
test = pd.read_csv(os.path.join(datasets_dir, "twitter_testBALANCED.csv"))

vectors_path = os.path.join(project_root, 'DATASETS', 'VECTORS', 'tfidf')
X_train_full, _, X_test_full, _ = vr.load_tfidf(prefix=vectors_path)
X_train = X_train_full[train.index]
X_test = X_test_full[test.index]
y_train = train['label']
y_test = test['label']

# Valores de C a probar
c_values = [0.001, 0.01, 0.09,0.2, 0.5, 1, 2, 10]

# Valores por defecto
params = {
    'penalty': 'l2',
    'loss': 'squared_hinge',
    'dual': False,
    'max_iter': 2000,
    'multi_class': 'ovr',
    'random_state': 42
}

train_acc = []
test_acc = []
failed = []
for c in c_values:
    params_c = params.copy()
    params_c['C'] = c
    try:
        model = LinearSVC(**params_c)
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        train_acc.append(accuracy_score(y_train, train_pred))
        test_acc.append(accuracy_score(y_test, test_pred))
    except Exception as e:
        train_acc.append(np.nan)
        test_acc.append(np.nan)
        failed.append((c, str(e)))

plt.figure(figsize=(8,5))
plt.plot(c_values, train_acc, marker='o', label='Train Accuracy')
plt.plot(c_values, test_acc, marker='s', label='Test Accuracy')
plt.xscale('log')
plt.xlabel('C (log scale)')
plt.ylabel('Accuracy')
plt.title('Evolución de la accuracy según C (LinearSVC)')
plt.legend()
plt.grid(True, which='both', axis='x')
plt.tight_layout()
plt.savefig(os.path.join(current_dir, 'svm_tuning_C.png'))
plt.close()
print('Guardada gráfica svm_tuning_C.png')
if failed:
    print(f"Valores fallidos para C: {failed}")
print('¡Tuning de C completado!')
