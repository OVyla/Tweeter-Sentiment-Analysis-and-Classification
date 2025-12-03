# Import necessary libraries
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.model_selection import GridSearchCV

# --- SUPPRESS WARNINGS ---
# Ignore future warnings to avoid cluttering the output
warnings.filterwarnings("ignore", category=FutureWarning)
# Ignore user warnings for the same reason
warnings.filterwarnings("ignore", category=UserWarning)

# --- RESOURCE CONFIGURATION ---
# Set a limit for the number of parallel jobs to run
N_JOBS_LIMIT = 8

def model_standard(X_train, y_train, **params):
    """
    Trains a standard Logistic Regression model.

    Args:
        X_train: Training data features.
        y_train: Training data labels.
        **params: Additional parameters for the LogisticRegression model.

    Returns:
        The trained Logistic Regression model.
    """
    # Set default parameters for the model
    default_params = {'solver': 'saga', 'n_jobs': N_JOBS_LIMIT, 'random_state': 42}
    # Combine default parameters with any user-provided parameters
    final_params = {**default_params, **params}
    
    # Initialize the Logistic Regression model with the final parameters
    model = LogisticRegression(**final_params)
    # Train the model on the training data
    model.fit(X_train, y_train)
    # Return the trained model
    return model

def model_one_vs_one(X_train, y_train, **params):
    """
    Trains a Logistic Regression model using the One-vs-One strategy for multiclass classification.

    Args:
        X_train: Training data features.
        y_train: Training data labels.
        **params: Additional parameters for the base LogisticRegression model.

    Returns:
        The trained OneVsOneClassifier model.
    """
    # Set default parameters for the base logistic regression
    default_params = {'solver': 'liblinear', 'random_state': 42}
    # Combine default parameters with any user-provided parameters
    final_params = {**default_params, **params}
    
    # Initialize the base Logistic Regression model
    base_lr = LogisticRegression(**final_params)
    # Initialize the One-vs-One classifier with the base model
    model = OneVsOneClassifier(base_lr, n_jobs=N_JOBS_LIMIT)
    # Train the model on the training data
    model.fit(X_train, y_train)
    # Return the trained model
    return model

def model_one_vs_rest(X_train, y_train, **params):
    """
    Trains a Logistic Regression model using the One-vs-Rest strategy for multiclass classification.

    Args:
        X_train: Training data features.
        y_train: Training data labels.
        **params: Additional parameters for the base LogisticRegression model.

    Returns:
        The trained OneVsRestClassifier model.
    """
    # Set default parameters for the base logistic regression
    default_params = {'solver': 'liblinear', 'random_state': 42}
    # Combine default parameters with any user-provided parameters
    final_params = {**default_params, **params}
    
    # Initialize the base Logistic Regression model
    base_lr = LogisticRegression(**final_params)
    # Initialize the One-vs-Rest classifier with the base model
    model = OneVsRestClassifier(base_lr, n_jobs=N_JOBS_LIMIT)
    # Train the model on the training data
    model.fit(X_train, y_train)
    # Return the trained model
    return model

def model_grid_search(X_train, y_train, **params):
    """
    Performs a grid search to find the best hyperparameters for a Logistic Regression model.

    Args:
        X_train: Training data features.
        y_train: Training data labels.
        **params: Additional parameters (not used in this function, but included for consistency).

    Returns:
        The best estimator found by GridSearchCV.
    """
    # Print a message indicating the start of the grid search
    print(f"Running GridSearchCV (max jobs={N_JOBS_LIMIT})...")
    
    # Initialize a base Logistic Regression model with n_jobs=1 to avoid a thread explosion
    # when GridSearchCV parallelizes the process.
    base_logreg = LogisticRegression(solver='saga', max_iter=1000, n_jobs=1, random_state=42)
    
    # Define the grid of hyperparameters to search over
    param_grid = {
        'C': [0.01, 0.1, 0.5, 1, 2, 5, 10],  # Regularization strength
        'class_weight': [None, 'balanced'],  # Weights for classes
        'penalty': ['l1', 'l2'],             # Regularization penalty
        'multi_class': ['auto', 'ovr', 'multinomial'] # Multiclass strategy
    }
    
    # Initialize GridSearchCV with the base model, parameter grid, and other settings
    gs = GridSearchCV(base_logreg, param_grid, cv=3, scoring='f1_macro', verbose=1, n_jobs=N_JOBS_LIMIT)
    # Fit GridSearchCV to the training data to find the best parameters
    gs.fit(X_train, y_train)
    
    # Print the best parameters found by the grid search
    print(f"Best parameters: {gs.best_params_}")
    # Return the model with the best found hyperparameters
    return gs.best_estimator_