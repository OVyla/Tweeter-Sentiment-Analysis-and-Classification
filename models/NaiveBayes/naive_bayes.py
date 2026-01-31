# Import necessary libraries from scikit-learn
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB, ComplementNB
from sklearn.model_selection import GridSearchCV

def model_complement(X_train, y_train, **params):
    """
    Trains a Complement Naive Bayes model.
    This variant is suitable for imbalanced datasets.
    """
    # Initialize the ComplementNB model with any provided parameters
    model = ComplementNB(**params)
    # Train the model on the training data
    model.fit(X_train, y_train)
    # Return the trained model
    return model

def model_multinomial(X_train, y_train, **params):
    """
    Trains a Multinomial Naive Bayes model.
    This is typically used for text classification with discrete features (e.g., word counts).
    """
    # Initialize the MultinomialNB model with any provided parameters
    model = MultinomialNB(**params)
    # Train the model on the training data
    model.fit(X_train, y_train)
    # Return the trained model
    return model

def model_bernoulli(X_train, y_train, **params):
    """
    Trains a Bernoulli Naive Bayes model.
    This is used for binary/boolean features (e.g., word presence/absence).
    """
    # Initialize the BernoulliNB model with any provided parameters
    model = BernoulliNB(**params)
    # Train the model on the training data
    model.fit(X_train, y_train)
    # Return the trained model
    return model

def model_gaussian(X_train, y_train, **params):
    """
    Trains a Gaussian Naive Bayes model.
    Note: GaussianNB assumes features follow a Gaussian distribution and requires dense arrays, not sparse matrices.
    """
    # Initialize the GaussianNB model with any provided parameters
    model = GaussianNB(**params)
    # Train the model on the training data
    model.fit(X_train, y_train)
    # Return the trained model
    return model

def model_grid_search(X_train, y_train, model_type='multinomial', **params):
    """
    Performs a GridSearchCV for Naive Bayes (MultinomialNB, BernoulliNB, and ComplementNB)
    to find the best hyperparameters.
    """
    # Select the model and parameter grid based on the specified model_type
    if model_type == 'multinomial':
        model = MultinomialNB()
        param_grid = {
            'alpha': [0.1, 0.4, 0.8, 1.0, 2.0, 5.0],  # Additive (Laplace/Lidstone) smoothing parameter
            'fit_prior': [True, False]               # Whether to learn class prior probabilities or use a uniform prior
        }
    elif model_type == 'bernoulli':
        model = BernoulliNB()
        param_grid = {
            'alpha': [0.1, 0.4, 0.8, 1.0, 2.0, 5.0],  # Smoothing parameter
            'fit_prior': [True, False],              # Whether to learn class prior probabilities
            'binarize': [0.0, 0.1, 0.2, 0.5]         # Threshold for binarizing features
        }
    elif model_type == 'complement':
        model = ComplementNB()
        param_grid = {
            'alpha': [0.1, 0.4, 0.8, 1.0, 2.0, 5.0],  # Smoothing parameter
            'fit_prior': [True, False]               # Whether to learn class prior probabilities
        }
    else:
        # Raise an error if an unsupported model type is provided
        raise ValueError('model_type must be "multinomial", "bernoulli", or "complement"')

    # Initialize GridSearchCV with the model, parameter grid, and other settings
    # It will use 3-fold cross-validation and optimize for the 'f1_macro' score.
    # n_jobs=-1 uses all available CPU cores.
    gs = GridSearchCV(model, param_grid, cv=3, scoring='f1_macro', verbose=1, n_jobs=-1)
    # Fit GridSearchCV to the data to find the best hyperparameters
    gs.fit(X_train, y_train)
    # Print the best parameters found
    print(f"Best parameters: {gs.best_params_}")
    # Return the best estimator (model with the best found hyperparameters)
    return gs.best_estimator_
