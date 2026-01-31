# Import necessary libraries
import pandas as pd
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier
)

# Try importing LightGBM, handle if missing
try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None # If LightGBM is not installed, set it to None

def lightgbm_multiclass(X_train, y_train, **params):
    """
    Trains a native LightGBM multiclass classifier.
    FIXED: Converts input to numpy array to prevent feature name warnings during prediction.
    """
    if LGBMClassifier is None:
        raise ImportError("LightGBM is not installed. Please install it with 'pip install lightgbm'.")
    
    # If the training data has a 'values' attribute (like a pandas DataFrame), convert it to a numpy array
    if hasattr(X_train, "values"):
        X_train = X_train.values  # Strip feature names by converting DataFrame to Numpy 

    # Automatically detect the number of classes from the training labels
    num_class = len(set(y_train))
    # Set default parameters for the LightGBM classifier
    default_params = {
        'random_state': 42,
        'n_estimators': 100,
        'force_col_wise': True,
        'objective': 'multiclass',
        'num_class': num_class
    }
    # Merge default parameters with any custom parameters provided
    final_params = {**default_params, **params}
    # Initialize the LightGBM classifier with the final parameters
    model = LGBMClassifier(**final_params)
    # Train the model
    model.fit(X_train, y_train)
    return model
# CatBoost OVR
def catboost_ovr(X_train, y_train, **params):
    """
    CatBoost One-vs-Rest para clasificación multiclase.
    """
    if CatBoostClassifier is None:
        raise ImportError("CatBoost no está instalado. Instala con 'pip install catboost'.")
    from sklearn.multiclass import OneVsRestClassifier
    default_params = {'random_state': 42, 'iterations': 100, 'verbose': 0}
    final_params = {**default_params, **params}
    base_model = CatBoostClassifier(**final_params)
    model = OneVsRestClassifier(base_model)
    model.fit(X_train, y_train)
    return model
# XGBoost multiclass
def xgboost_multiclass(X_train, y_train, **params):
    """
    XGBoost nativo multiclass (objective='multi:softprob') para clasificación multiclase.
    Convierte etiquetas string a números automáticamente.
    """
    if XGBClassifier is None:
        raise ImportError("XGBoost no está instalado. Instala con 'pip install xgboost'.")
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    num_class = len(le.classes_)
    default_params = {
        'random_state': 42,
        'n_estimators': 100,
        'objective': 'multi:softprob',
        'num_class': num_class,
        'use_label_encoder': False,
        'eval_metric': 'mlogloss'
    }
    final_params = {**default_params, **params}
    model = XGBClassifier(**final_params)
    model.fit(X_train, y_train_enc)
    # Adjuntar el label encoder al modelo para decodificar luego
    model._label_encoder = le
    return model
def lightgbm_ovr(X_train, y_train, **params):
    """
    Trains a LightGBM One-vs-Rest classifier for multiclass classification.
    FIXED: Converts input to numpy array to prevent feature name warnings during prediction.
    """
    if LGBMClassifier is None:
        raise ImportError("LightGBM is not installed. Please install it with 'pip install lightgbm'.")
    
    # If the training data has a 'values' attribute, convert it to a numpy array
    if hasattr(X_train, "values"):
        X_train = X_train.values  # FIX: Strip feature names by converting DataFrame to Numpy 

    # Set default parameters for the base LightGBM model
    default_params = {'random_state': 42, 'n_estimators': 100, 'force_col_wise': True}
    # Merge default parameters with any custom parameters
    final_params = {**default_params, **params}
    # Initialize the base LightGBM model
    base_model = LGBMClassifier(**final_params)
    # Wrap the base model in a One-vs-Rest classifier
    model = OneVsRestClassifier(base_model)
    # Train the model
    model.fit(X_train, y_train)
    return model

def ada_boost_ovr(X_train, y_train, **params):
    """
    Trains an AdaBoost One-vs-Rest classifier for multiclass classification.
    """
    # Set default parameters for the base AdaBoost model
    default_params = {'random_state': 42, 'n_estimators': 100}
    # Merge default parameters with any custom parameters
    final_params = {**default_params, **params}
    # Initialize the base AdaBoost model
    base_model = AdaBoostClassifier(**final_params)
    # Wrap the base model in a One-vs-Rest classifier
    model = OneVsRestClassifier(base_model)
    # Train the model
    model.fit(X_train, y_train)
    return model

def extra_trees_ovr(X_train, y_train, **params):
    """
    Trains an ExtraTrees One-vs-Rest classifier for multiclass classification.
    """
    # Set default parameters for the base ExtraTrees model, using all available CPU cores
    default_params = {'random_state': 42, 'n_jobs': -1}
    # Merge default parameters with any custom parameters
    final_params = {**default_params, **params}
    # Initialize the base ExtraTrees model
    base_model = ExtraTreesClassifier(**final_params)
    # Wrap the base model in a One-vs-Rest classifier
    model = OneVsRestClassifier(base_model)
    # Train the model
    model.fit(X_train, y_train)
    return model

def gradient_boosting_ovr(X_train, y_train, **params):
    """
    Trains a Gradient Boosting One-vs-Rest classifier for multiclass classification.
    """
    # Set default parameters for the base Gradient Boosting model
    default_params = {'random_state': 42}
    # Merge default parameters with any custom parameters
    final_params = {**default_params, **params}
    # Initialize the base Gradient Boosting model
    base_model = GradientBoostingClassifier(**final_params)
    # Wrap the base model in a One-vs-Rest classifier
    model = OneVsRestClassifier(base_model)
    # Train the model
    model.fit(X_train, y_train)
    return model

def ada_boost(X_train, y_train, **params):
    """
    Trains a standard AdaBoost classifier for multiclass classification.
    """
    # Set default parameters for the AdaBoost model
    default_params = {'random_state': 42, 'n_estimators': 100}
    # Merge default parameters with any custom parameters
    final_params = {**default_params, **params}
    # Initialize the AdaBoost model
    model = AdaBoostClassifier(**final_params)
    # Train the model
    model.fit(X_train, y_train)
    return model

def extra_trees(X_train, y_train, **params):
    """
    Trains a standard ExtraTrees classifier for multiclass classification.
    """
    # Set default parameters for the ExtraTrees model, using all available CPU cores
    default_params = {'random_state': 42, 'n_jobs': -1}
    # Merge default parameters with any custom parameters
    final_params = {**default_params, **params}
    # Initialize the ExtraTrees model
    model = ExtraTreesClassifier(**final_params)
    # Train the model
    model.fit(X_train, y_train)
    return model

def gradient_boosting(X_train, y_train, **params):
    """
    Trains a standard Gradient Boosting classifier for multiclass classification.
    """
    # Set default parameters for the Gradient Boosting model
    default_params = {'random_state': 42}
    # Merge default parameters with any custom parameters
    final_params = {**default_params, **params}
    # Initialize the Gradient Boosting model
    model = GradientBoostingClassifier(**final_params)
    # Train the model
    model.fit(X_train, y_train)
    return model

def rf_standard(X_train, y_train, **params):
    """
    Trains a standard Random Forest classifier for multiclass classification.
    """
    # Set default parameters for the Random Forest model, using all available CPU cores
    default_params = {'n_estimators': 100, 'random_state': 42, 'n_jobs': -1}
    # Merge default parameters with any custom parameters
    final_params = {**default_params, **params}
    # Initialize the Random Forest model
    model = RandomForestClassifier(**final_params)
    # Train the model
    model.fit(X_train, y_train)
    return model

def rf_one_vs_rest(X_train, y_train, **params):
    """
    Trains a Random Forest One-vs-Rest classifier for multiclass classification.
    """
    # Set default parameters for the base Random Forest model, using all available CPU cores
    default_params = {'n_estimators': 100, 'random_state': 42, 'n_jobs': -1}
    # Merge default parameters with any custom parameters
    final_params = {**default_params, **params}
    # Initialize the base Random Forest model
    base_rf = RandomForestClassifier(**final_params)
    # Wrap the base model in a One-vs-Rest classifier
    model = OneVsRestClassifier(base_rf)
    # Train the model
    model.fit(X_train, y_train)
    return model

def rf_one_vs_one(X_train, y_train, **params):
    """
    Trains a Random Forest One-vs-One classifier for multiclass classification.
    """
    # Set default parameters for the base Random Forest model, using all available CPU cores
    default_params = {'n_estimators': 100, 'random_state': 42, 'n_jobs': -1}
    # Merge default parameters with any custom parameters
    final_params = {**default_params, **params}
    # Initialize the base Random Forest model
    base_rf = RandomForestClassifier(**final_params)
    # Wrap the base model in a One-vs-One classifier
    model = OneVsOneClassifier(base_rf)
    # Train the model
    model.fit(X_train, y_train)

    return model
