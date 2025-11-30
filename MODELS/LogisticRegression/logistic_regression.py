from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.model_selection import GridSearchCV

def model_standard(X_train, y_train, **params):
    # 'saga' suporta n_jobs internament, ideal per multiclass='multinomial'
    default_params = {'solver': 'saga', 'n_jobs': -1, 'random_state': 42}
    final_params = {**default_params, **params}
    
    model = LogisticRegression(**final_params)
    model.fit(X_train, y_train)
    return model

def model_one_vs_one(X_train, y_train, **params):
    # Mantenim liblinear (ràpid per sparse), però paral·leliitzem el wrapper
    default_params = {'solver': 'liblinear', 'random_state': 42}
    final_params = {**default_params, **params}
    
    base_lr = LogisticRegression(**final_params)
    # AFEGIT: n_jobs=-1 per entrenar els parells de classes en paral·lel
    model = OneVsOneClassifier(base_lr, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

def model_one_vs_rest(X_train, y_train, **params):
    default_params = {'solver': 'liblinear', 'random_state': 42}
    final_params = {**default_params, **params}
    
    base_lr = LogisticRegression(**final_params)
    # AFEGIT: n_jobs=-1 per entrenar cada classe vs la resta en paral·lel
    model = OneVsRestClassifier(base_lr, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

def model_grid_search(X_train, y_train, **params):
    print("Executant GridSearchCV...")
    base_logreg = LogisticRegression(solver='saga', max_iter=1000, n_jobs=-1, random_state=42)
    
    param_grid = {
        'C': [0.01, 0.1, 0.5, 1, 2, 5, 10],
        'class_weight': [None, 'balanced'],
        'penalty': ['l1', 'l2'],
        'multi_class': ['auto', 'ovr', 'multinomial']
    }
    
    # n_jobs=-1 ja hi era aquí, aquest hauria de funcionar bé
    gs = GridSearchCV(base_logreg, param_grid, cv=3, scoring='f1_macro', verbose=1, n_jobs=-1)
    gs.fit(X_train, y_train)
    print(f"Millors paràmetres: {gs.best_params_}")
    return gs.best_estimator_