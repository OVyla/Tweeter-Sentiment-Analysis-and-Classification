import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from vector_representation import load_tfidf

# Cargar datos
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
train = pd.read_csv(os.path.join(base_dir, 'twitter_trainBALANCED.csv')).sample(frac=0.1, random_state=42)
val = pd.read_csv(os.path.join(base_dir, 'twitter_valBALANCED.csv')).sample(frac=0.1, random_state=42)
test = pd.read_csv(os.path.join(base_dir, 'twitter_testBALANCED.csv')).sample(frac=0.1, random_state=42)
X_train, X_val, X_test, _ = load_tfidf(prefix="./VECTORES/tfidf")
X_train = X_train[train.index]
X_val = X_val[val.index]
X_test = X_test[test.index]
y_train = train['label'].reset_index(drop=True)
y_val = val['label'].reset_index(drop=True)
y_test = test['label'].reset_index(drop=True)

# Parámetros a probar
param_grid = {
    'n_estimators': [5,10, 50, 100, 200,300],
    'max_depth': [5, 10, 20, 50,100, None],
    'min_samples_leaf': [ 5, 10, 20,50,100,200],
    'min_samples_split': [5, 10, 20,60,100],
    'max_features': ['sqrt', 'log2', None],
    'criterion': ['gini', 'entropy', 'log_loss'],
    'bootstrap': [True, False],
}

results = []
for param, values in param_grid.items():
    print(f"\nProbando parámetro: {param}")
    for v in values:
        kwargs = {
            'n_estimators': 100,
            'max_depth': 20,
            'min_samples_leaf': 1,
            'min_samples_split': 2,
            'max_features': 'sqrt',
            'criterion': 'gini',
            'bootstrap': True,
            'random_state': 42,
            'n_jobs': -1
        }
        kwargs[param] = v
        try:
            clf = RandomForestClassifier(**kwargs)
            clf.fit(X_train, y_train)
            train_preds = clf.predict(X_train)
            val_preds = clf.predict(X_val)
            test_preds = clf.predict(X_test)
            acc_train = accuracy_score(y_train, train_preds)
            acc_val = accuracy_score(y_val, val_preds)
            acc_test = accuracy_score(y_test, test_preds)
            print(f"{param}={v}: Train={acc_train:.4f} | Val={acc_val:.4f} | Test={acc_test:.4f}")
            results.append((param, v, acc_train, acc_val, acc_test))
        except Exception as e:
            print(f"Error con {param}={v}: {e}")

# Guardar resultados
results_df = pd.DataFrame(results, columns=['param', 'value', 'train_accuracy', 'val_accuracy', 'test_accuracy'])
results_df.to_csv('random_forest_param_individual_results.csv', index=False)
print('Resultados guardados en random_forest_param_individual_results.csv')

# Graficar resultados
import matplotlib.pyplot as plt
cat_params = ['max_features', 'criterion', 'bootstrap']
num_params = ['n_estimators', 'max_depth', 'min_samples_leaf', 'min_samples_split']

for param in cat_params:
    df = results_df[results_df['param'] == param]
    plt.figure(figsize=(8,5))
    x = df['value'].astype(str)
    width = 0.35
    idx = range(len(x))
    plt.bar([i - width/2 for i in idx], df['train_accuracy'], width=width, color='blue', alpha=0.7, label='Train')
    plt.bar([i + width/2 for i in idx], df['test_accuracy'], width=width, color='orange', alpha=0.7, label='Test')
    plt.xlabel(param)
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy según {param} (RandomForest)')
    plt.ylim(0, 1)
    plt.xticks(idx, x)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'rf_bar_{param}_train_test.png')
    plt.close()
    print(f'Guardada gráfica de barras para {param}: rf_bar_{param}_train_test.png')

for param in num_params:
    df = results_df[results_df['param'] == param]
    plt.figure(figsize=(8,5))
    plt.plot(df['value'], df['train_accuracy'], marker='o', color='blue', label='Train')
    plt.plot(df['value'], df['test_accuracy'], marker='s', color='orange', label='Test')
    plt.xlabel(param)
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy según {param} (RandomForest)')
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'rf_line_{param}_train_test.png')
    plt.close()
    print(f'Guardada gráfica de línea para {param}: rf_line_{param}_train_test.png')
