from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier
def lightgbm_multiclass(X_train, y_train, **params):
    """
    LightGBM nativo multiclass (objective='multiclass') para clasificación multiclase.
    """
    if LGBMClassifier is None:
        raise ImportError("LightGBM no está instalado. Instala con 'pip install lightgbm'.")
    # Detectar número de clases automáticamente
    num_class = len(set(y_train))
    default_params = {
        'random_state': 42,
        'n_estimators': 100,
        'force_col_wise': True,
        'objective': 'multiclass',
        'num_class': num_class
    }
    final_params = {**default_params, **params}
    model = LGBMClassifier(**final_params)
    model.fit(X_train, y_train)
    return model

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

def lightgbm_ovr(X_train, y_train, **params):
    """
    LightGBM One-vs-Rest para clasificación multiclase.
    """
    if LGBMClassifier is None:
        raise ImportError("LightGBM no está instalado. Instala con 'pip install lightgbm'.")
    from sklearn.multiclass import OneVsRestClassifier
    default_params = {'random_state': 42, 'n_estimators': 100, 'force_col_wise': True}
    final_params = {**default_params, **params}
    base_model = LGBMClassifier(**final_params)
    model = OneVsRestClassifier(base_model)
    model.fit(X_train, y_train)
    return model

def ada_boost_ovr(X_train, y_train, **params):
    """
    AdaBoost One-vs-Rest para clasificación multiclase.
    """
    default_params = {'random_state': 42, 'n_estimators': 100}
    final_params = {**default_params, **params}
    base_model = AdaBoostClassifier(**final_params)
    model = OneVsRestClassifier(base_model)
    model.fit(X_train, y_train)
    return model

def extra_trees_ovr(X_train, y_train, **params):
    """
    ExtraTrees One-vs-Rest para clasificación multiclase.
    """
    default_params = {'random_state': 42, 'n_jobs': -1}
    final_params = {**default_params, **params}
    base_model = ExtraTreesClassifier(**final_params)
    model = OneVsRestClassifier(base_model)
    model.fit(X_train, y_train)
    return model

def gradient_boosting_ovr(X_train, y_train, **params):
    """
    Gradient Boosting One-vs-Rest para clasificación multiclase.
    """
    default_params = {'random_state': 42}
    final_params = {**default_params, **params}
    base_model = GradientBoostingClassifier(**final_params)
    model = OneVsRestClassifier(base_model)
    model.fit(X_train, y_train)
    return model


def ada_boost(X_train, y_train, **params):
    """
    AdaBoost para clasificación multiclase.
    """
    default_params = {'random_state': 42, 'n_estimators': 100}
    final_params = {**default_params, **params}
    model = AdaBoostClassifier(**final_params)
    model.fit(X_train, y_train)
    return model

def extra_trees(X_train, y_train, **params):
    """
    ExtraTrees para clasificación multiclase.
    """
    default_params = {'random_state': 42, 'n_jobs': -1}
    final_params = {**default_params, **params}
    model = ExtraTreesClassifier(**final_params)
    model.fit(X_train, y_train)
    return model

def gradient_boosting(X_train, y_train, **params):
    """
    Gradient Boosting para clasificación multiclase.
    """
    default_params = {'random_state': 42}
    final_params = {**default_params, **params}
    model = GradientBoostingClassifier(**final_params)
    model.fit(X_train, y_train)
    return model


def rf_standard(X_train, y_train, **params):
    """
    Random Forest estándar para clasificación multiclase.
    """
    default_params = {'n_estimators': 100, 'random_state': 42, 'n_jobs': -1}
    final_params = {**default_params, **params}
    model = RandomForestClassifier(**final_params)
    model.fit(X_train, y_train)
    return model

def rf_one_vs_rest(X_train, y_train, **params):
    """
    Random Forest One-vs-Rest para clasificación multiclase.
    """
    default_params = {'n_estimators': 100, 'random_state': 42, 'n_jobs': -1}
    final_params = {**default_params, **params}
    base_rf = RandomForestClassifier(**final_params)
    model = OneVsRestClassifier(base_rf)
    model.fit(X_train, y_train)
    return model

def rf_one_vs_one(X_train, y_train, **params):
    """
    Random Forest One-vs-One para clasificación multiclase.
    """
    default_params = {'n_estimators': 100, 'random_state': 42, 'n_jobs': -1}
    final_params = {**default_params, **params}
    base_rf = RandomForestClassifier(**final_params)
    model = OneVsOneClassifier(base_rf)
    model.fit(X_train, y_train)
    return model
