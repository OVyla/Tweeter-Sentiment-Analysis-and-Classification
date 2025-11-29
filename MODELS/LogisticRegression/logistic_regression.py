from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.model_selection import GridSearchCV

def model_standard(X_train, y_train, **params):
    # Valors per defecte si no s'especifiquen a params
    default_params = {'solver': 'saga', 'n_jobs': -1, 'random_state': 42}
    # Actualitzem defectes amb els paràmetres que arriben del run.py
    final_params = {**default_params, **params}
    
    model = LogisticRegression(**final_params)
    model.fit(X_train, y_train)
    return model

def model_one_vs_one(X_train, y_train, **params):
    default_params = {'solver': 'liblinear', 'random_state': 42}
    final_params = {**default_params, **params}
    
    base_lr = LogisticRegression(**final_params)
    model = OneVsOneClassifier(base_lr)
    model.fit(X_train, y_train)
    return model

def model_one_vs_rest(X_train, y_train, **params):
    default_params = {'solver': 'liblinear', 'random_state': 42}
    final_params = {**default_params, **params}
    
    base_lr = LogisticRegression(**final_params)
    model = OneVsRestClassifier(base_lr)
    model.fit(X_train, y_train)
    return model

def model_grid_search(X_train, y_train, **params):
    # GridSearch ignora els hiperparàmetres globals simples perquè busca els seus propis,
    # però podem permetre configurar el grid si calgués. Aquí el deixem fix.
    print("Executant GridSearchCV...")
    base_logreg = LogisticRegression(solver='saga', max_iter=1000, n_jobs=-1, random_state=42)
    
    param_grid = {
        'C': [0.01, 0.1, 0.5, 1, 2, 5, 10],
        'class_weight': [None, 'balanced'],
        'penalty': ['l1', 'l2'],
        'multi_class': ['auto', 'ovr', 'multinomial']
    }
    
    gs = GridSearchCV(base_logreg, param_grid, cv=3, scoring='f1_macro', verbose=1, n_jobs=-1)
    gs.fit(X_train, y_train)
    print(f"Millors paràmetres: {gs.best_params_}")
    return gs.best_estimator_
