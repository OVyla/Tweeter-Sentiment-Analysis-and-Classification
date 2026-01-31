import os
import pandas as pd
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..', '..'))
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'SPLIT')

# !!! CRITICAL: SVM is slow. We use a small sample for Tuning !!!
# 20,000 is enough to find the best 'C' and 'kernel'
TRAIN_SAMPLE_SIZE = 20000  

def load_data_subset():
    print("Loading datasets for SVM Tuning...")
    train_path = os.path.join(DATA_PATH, 'twitter_trainBALANCED.csv')
    val_path = os.path.join(DATA_PATH, 'twitter_valBALANCED.csv')

    df_train = pd.read_csv(train_path).dropna(subset=['text', 'label'])
    df_val = pd.read_csv(val_path).dropna(subset=['text', 'label'])

    # Strict sampling for SVM speed
    if len(df_train) > TRAIN_SAMPLE_SIZE:
        print(f"Sampling {TRAIN_SAMPLE_SIZE} rows to ensure SVM finishes quickly...")
        df_train = df_train.sample(n=TRAIN_SAMPLE_SIZE, random_state=42)
    
    # Keep validation small too for quick scoring
    if len(df_val) > 5000:
        df_val = df_val.sample(n=5000, random_state=42)
        
    return df_train, df_val

def run_svm_tuning():
    start_time = time.time()
    df_train, df_val = load_data_subset()

    X_train = df_train['text']
    y_train = df_train['label']
    X_val = df_val['text']
    y_val = df_val['label']

    # --- 1. DEFINE PIPELINE ---
    print("\nSetting up SVM Pipeline...")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        # cache_size=1000 MB helps speed it up
        ('svm', SVC(random_state=42, cache_size=1000)) 
    ])

    # --- 2. DEFINE PARAMETERS ---
    # We focus on C and Kernel. 
    # 'linear' is usually best for text, but 'rbf' captures non-linear relations.
    param_dist = {
        'svm__C': [0.1, 1, 10, 100],
        'svm__kernel': ['linear', 'rbf'], 
        'svm__gamma': ['scale', 'auto'] # Only relevant for rbf
    }

    print(f"Starting RandomizedSearchCV with parameters: {param_dist}")
    
    # n_iter=10 keeps it fast. We don't need to try every single combination.
    random_search = RandomizedSearchCV(
        pipeline, 
        param_distributions=param_dist, 
        n_iter=10, 
        cv=3, 
        scoring='f1_weighted', 
        verbose=2,
        n_jobs=-1,
        random_state=42
    )

    random_search.fit(X_train, y_train)

    # --- 3. RESULTS ---
    print("\n--- Best SVM Parameters ---")
    print(random_search.best_params_)
    print(f"Best CV F1-Score: {random_search.best_score_:.4f}")

    best_model = random_search.best_estimator_
    print("\nEvaluating best model on Validation subset...")
    y_pred = best_model.predict(X_val)
    print(classification_report(y_val, y_pred))
    
    minutes = (time.time() - start_time) / 60
    print(f"Total execution time: {minutes:.2f} minutes")

if __name__ == "__main__":
    run_svm_tuning()


"""
OUTPUT D'AQUEST CODI:

--- Best SVM Parameters ---
{'svm__kernel': 'rbf', 'svm__gamma': 'scale', 'svm__C': 10}
Best CV F1-Score: 0.6977

Evaluating best model on Validation subset...
              precision    recall  f1-score   support

    negative       0.74      0.71      0.72      1710
     neutral       0.63      0.70      0.67      1646
    positive       0.75      0.70      0.72      1644

    accuracy                           0.70      5000
   macro avg       0.71      0.70      0.70      5000
weighted avg       0.71      0.70      0.70      5000

Total execution time: 3.04 minutes

"""