import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, f1_score
import time

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..', '..'))
DATA_PATH = os.path.join(PROJECT_ROOT, 'DATASETS', 'SPLIT')
OUTPUT_PLOT_DIR = os.path.join(PROJECT_ROOT, 'GRAFIQUES', 'RandomForest', 'Optimization')
os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)

# !!! OPTIMIZATION: TRAIN ON A SUBSET !!!
# 100,000 rows is usually enough to find the best hyperparameters.
# Once found, you apply them to the full 1M dataset in a separate training run.
TRAIN_SAMPLE_SIZE = 100000  

def load_data():
    """Loads Training and Validation data."""
    print("Loading datasets...")
    train_path = os.path.join(DATA_PATH, 'twitter_trainBALANCED.csv')
    val_path = os.path.join(DATA_PATH, 'twitter_valBALANCED.csv')

    df_train = pd.read_csv(train_path).dropna(subset=['text', 'label'])
    df_val = pd.read_csv(val_path).dropna(subset=['text', 'label'])

    print(f"Original Train shape: {df_train.shape}")
    print(f"Val shape: {df_val.shape}")

    # --- SAMPLING FOR TUNING ---
    if len(df_train) > TRAIN_SAMPLE_SIZE:
        print(f"⚠️  Sampling {TRAIN_SAMPLE_SIZE} rows for Hyperparameter Tuning to save time...")
        df_train = df_train.sample(n=TRAIN_SAMPLE_SIZE, random_state=42)
    
    return df_train, df_val

def run_tuning_and_analysis():
    start_time = time.time()
    df_train, df_val = load_data()

    X_train = df_train['text']
    y_train = df_train['label']
    X_val = df_val['text']
    y_val = df_val['label']

    # --- 1. DEFINE PIPELINE ---
    print("\nSetting up Pipeline...")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)), 
        # n_jobs=-1 uses all CPU cores
        ('rf', RandomForestClassifier(random_state=42, n_jobs=-1)) 
    ])

    # --- 2. DEFINE HYPERPARAMETER DISTRIBUTION ---
    # Using RandomizedSearch allows us to explore a wider range efficiently
    param_dist = {
        'rf__n_estimators': [50, 100, 200, 300], 
        'rf__max_depth': [None, 20, 50, 100],
        'rf__min_samples_leaf': [1, 5, 10]
    }

    print(f"Starting RandomizedSearchCV with parameters: {param_dist}")
    
    # n_iter=15 means we randomly try 15 combinations, not all of them.
    # This reduces fits from 90 down to 45 (15 * 3 folds)
    random_search = RandomizedSearchCV(
        pipeline, 
        param_distributions=param_dist, 
        n_iter=15, 
        cv=3, 
        scoring='f1_weighted', 
        verbose=2,
        n_jobs=-1,
        random_state=42
    )

    random_search.fit(X_train, y_train)

    # --- 3. RESULTS ANALYSIS ---
    print("\n--- Best Parameters Found (on Subset) ---")
    print(random_search.best_params_)
    print(f"Best CV F1-Score: {random_search.best_score_:.4f}")

    # Validation Evaluation with Best Model
    best_model = random_search.best_estimator_
    print("Evaluating best model on Validation Set...")
    y_pred = best_model.predict(X_val)
    print("\n--- Validation Performance (Best Model) ---")
    print(classification_report(y_val, y_pred))

    # --- 4. SENSITIVITY PLOT GENERATION ---
    results_df = pd.DataFrame(random_search.cv_results_)
    
    # Sort by score to see trend
    sensitivity_data = results_df.sort_values(by='param_rf__n_estimators')

    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # We plot a scatter plot because RandomizedSearch doesn't have a perfect grid
    sns.lineplot(
        data=sensitivity_data,
        x='param_rf__n_estimators', 
        y='mean_test_score',
        marker='o',
        estimator='mean', # Determine mean if multiple points exist for same n_estimators
        errorbar=None,
        color='#2c3e50',
        label='Weighted F1 Score'
    )

    plt.title(f'Random Forest Sensitivity: n_estimators (Tuned on {TRAIN_SAMPLE_SIZE} samples)', fontsize=14)
    plt.xlabel('Number of Estimators (Trees)', fontsize=12)
    plt.ylabel('Mean CV F1-Score (Weighted)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # Save Plot
    plot_filename = 'rf_sensitivity_n_estimators_optimized.png'
    plot_path = os.path.join(OUTPUT_PLOT_DIR, plot_filename)
    plt.savefig(plot_path)
    print(f"\nPlot saved to: {plot_path}")
    plt.close()
    
    minutes = (time.time() - start_time) / 60
    print(f"Total execution time: {minutes:.2f} minutes")

if __name__ == "__main__":
    run_tuning_and_analysis()



"""
OUTPUT D'AQUEST CODI:

--- Best Parameters Found (on Subset) ---
{'rf__n_estimators': 300, 'rf__min_samples_leaf': 1, 'rf__max_depth': None}  <-- MILLORS HIPERPARÀMETRES
Best CV F1-Score: 0.6958
Evaluating best model on Validation Set...

--- Validation Performance (Best Model) ---
              precision    recall  f1-score   support

    negative       0.70      0.73      0.71     46409
     neutral       0.64      0.70      0.67     45157
    positive       0.79      0.67      0.72     45296

    accuracy                           0.70    136862
   macro avg       0.71      0.70      0.70    136862
weighted avg       0.71      0.70      0.70    136862

"""