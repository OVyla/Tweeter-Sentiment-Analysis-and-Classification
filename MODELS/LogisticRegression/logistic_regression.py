from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.model_selection import GridSearchCV

# --- CONFIGURACIÓ DE RECURSOS ---
# Canvia aquest número segons la teva CPU/RAM.
# 2 és conservador i segur. Si tens un PC potent, prova amb 4.
# Evita -1 si tens problemes d'estabilitat.
N_JOBS_LIMIT = 4 

def model_standard(X_train, y_train, **params):
    # 'saga' suporta n_jobs. Limitem amb la variable global.
    default_params = {'solver': 'saga', 'n_jobs': N_JOBS_LIMIT, 'random_state': 42}
    final_params = {**default_params, **params}
    
    model = LogisticRegression(**final_params)
    model.fit(X_train, y_train)
    return model

def model_one_vs_one(X_train, y_train, **params):
    default_params = {'solver': 'liblinear', 'random_state': 42}
    final_params = {**default_params, **params}
    
    base_lr = LogisticRegression(**final_params)
    
    # Limitem els jobs del wrapper
    model = OneVsOneClassifier(base_lr, n_jobs=N_JOBS_LIMIT)
    model.fit(X_train, y_train)
    return model

def model_one_vs_rest(X_train, y_train, **params):
    default_params = {'solver': 'liblinear', 'random_state': 42}
    final_params = {**default_params, **params}
    
    base_lr = LogisticRegression(**final_params)
    
    # Limitem els jobs del wrapper
    model = OneVsRestClassifier(base_lr, n_jobs=N_JOBS_LIMIT)
    model.fit(X_train, y_train)
    return model

def model_grid_search(X_train, y_train, **params):
    print(f"Executant GridSearchCV (max jobs={N_JOBS_LIMIT})...")
    
    # IMPORTANT: El model base ha de tenir n_jobs=1.
    # Si posem paral·lelisme aquí I al GridSearchCV, es multipliquen els fils
    # (Ex: 4 jobs al grid * 4 jobs al model = 16 fils) -> Això fa petar la RAM.
    base_logreg = LogisticRegression(solver='saga', max_iter=1000, n_jobs=1, random_state=42)
    
    param_grid = {
        'C': [0.01, 0.1, 0.5, 1, 2, 5, 10],
        'class_weight': [None, 'balanced'],
        'penalty': ['l1', 'l2'],
        'multi_class': ['auto', 'ovr', 'multinomial']
    }
    
    # El paral·lelisme el gestiona només el GridSearchCV
    gs = GridSearchCV(base_logreg, param_grid, cv=3, scoring='f1_macro', verbose=1, n_jobs=N_JOBS_LIMIT)
    gs.fit(X_train, y_train)
    print(f"Millors paràmetres: {gs.best_params_}")
    return gs.best_estimator_