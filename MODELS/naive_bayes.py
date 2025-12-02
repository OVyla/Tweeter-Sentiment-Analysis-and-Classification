from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB, ComplementNB

def model_complement(X_train, y_train, **params):
    """
    Entrena un modelo Naive Bayes ComplementNB.
    """
    model = ComplementNB(**params)
    model.fit(X_train, y_train)
    return model
from sklearn.model_selection import GridSearchCV


def model_multinomial(X_train, y_train, **params):
    """
    Entrena un modelo Naive Bayes MultinomialNB.
    """
    model = MultinomialNB(**params)
    model.fit(X_train, y_train)
    return model


def model_bernoulli(X_train, y_train, **params):
    """
    Entrena un modelo Naive Bayes BernoulliNB.
    """
    model = BernoulliNB(**params)
    model.fit(X_train, y_train)
    return model


def model_gaussian(X_train, y_train, **params):
    """
    Entrena un modelo Naive Bayes GaussianNB.
    Nota: GaussianNB requiere arrays densos, no matrices dispersas.
    """
    model = GaussianNB(**params)
    model.fit(X_train, y_train)
    return model


def model_grid_search(X_train, y_train, model_type='multinomial', **params):
    """
    Realiza GridSearchCV para Naive Bayes (MultinomialNB, BernoulliNB y ComplementNB).
    """
    if model_type == 'multinomial':
        model = MultinomialNB()
        param_grid = {
            'alpha': [0.1, 0.4, 0.8, 1.0, 2.0, 5.0],
            'fit_prior': [True, False]
        }
    elif model_type == 'bernoulli':
        model = BernoulliNB()
        param_grid = {
            'alpha': [0.1, 0.4, 0.8, 1.0, 2.0, 5.0],
            'fit_prior': [True, False],
            'binarize': [0.0, 0.1, 0.2, 0.5]
        }
    elif model_type == 'complement':
        model = ComplementNB()
        param_grid = {
            'alpha': [0.1, 0.4, 0.8, 1.0, 2.0, 5.0],
            'fit_prior': [True, False]
        }
    else:
        raise ValueError('model_type debe ser "multinomial", "bernoulli" o "complement"')

    gs = GridSearchCV(model, param_grid, cv=3, scoring='f1_macro', verbose=1, n_jobs=-1)
    gs.fit(X_train, y_train)
    print(f"Mejores parámetros: {gs.best_params_}")
    return gs.best_estimator_
