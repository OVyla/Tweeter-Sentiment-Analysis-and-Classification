# Import necessary classes from scikit-learn
from sklearn.svm import LinearSVC, SVC
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier

def svm_standard(X_train, y_train, **params):
    """
    Trains a standard Linear Support Vector Machine (SVM) classifier.
    LinearSVC is generally faster for large datasets than SVC with a linear kernel.
    """
    # Set default parameters for the LinearSVC model
    # 'dual=False' is recommended when n_samples > n_features.
    # 'max_iter' is increased to help with convergence.
    default_params = {'dual': False, 'random_state': 42, 'max_iter': 2000}
    # Merge default parameters with any custom parameters provided
    final_params = {**default_params, **params}
    # Initialize the LinearSVC model with the final parameters
    model = LinearSVC(**final_params)
    # Train the model on the training data
    model.fit(X_train, y_train)
    # Return the trained model
    return model

def svm_one_vs_one(X_train, y_train, **params):
    """
    Trains a Linear SVM using the One-vs-One (OvO) strategy for multiclass classification.
    This trains N * (N - 1) / 2 classifiers, where N is the number of classes.
    """
    # Set default parameters for the base LinearSVC model
    default_params = {'dual': False, 'random_state': 42, 'max_iter': 2000}
    # Merge default parameters with any custom parameters
    final_params = {**default_params, **params}
    # Initialize the base LinearSVC model
    base_svc = LinearSVC(**final_params)
    # Wrap the base model in a OneVsOneClassifier
    model = OneVsOneClassifier(base_svc)
    # Train the model on the training data
    model.fit(X_train, y_train)
    # Return the trained model
    return model

def svm_one_vs_rest(X_train, y_train, **params):
    """
    Trains a Linear SVM using the One-vs-Rest (OvR) strategy for multiclass classification.
    This trains one classifier per class.
    """
    # Set default parameters for the base LinearSVC model
    default_params = {'dual': False, 'random_state': 42, 'max_iter': 2000}
    # Merge default parameters with any custom parameters
    final_params = {**default_params, **params}
    # Initialize the base LinearSVC model
    base_svc = LinearSVC(**final_params)
    # Wrap the base model in a OneVsRestClassifier
    model = OneVsRestClassifier(base_svc)
    # Train the model on the training data
    model.fit(X_train, y_train)
    # Return the trained model
    return model
