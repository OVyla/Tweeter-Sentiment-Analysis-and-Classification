
from sklearn.svm import LinearSVC, SVC
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier

def svm_standard(X_train, y_train, **params):
    default_params = {'dual': False, 'random_state': 42, 'max_iter': 2000}
    final_params = {**default_params, **params}
    model = LinearSVC(**final_params)
    model.fit(X_train, y_train)
    return model

def svm_one_vs_one(X_train, y_train, **params):
    default_params = {'dual': False, 'random_state': 42, 'max_iter': 2000}
    final_params = {**default_params, **params}
    base_svc = LinearSVC(**final_params)
    model = OneVsOneClassifier(base_svc)
    model.fit(X_train, y_train)
    return model

def svm_one_vs_rest(X_train, y_train, **params):
    default_params = {'dual': False, 'random_state': 42, 'max_iter': 2000}
    final_params = {**default_params, **params}
    base_svc = LinearSVC(**final_params)
    model = OneVsRestClassifier(base_svc)
    model.fit(X_train, y_train)
    return model

