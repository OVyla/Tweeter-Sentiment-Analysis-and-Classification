

import pandas as pd
import joblib
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))

if project_root not in sys.path:
    sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'AnalizarLimpiarDividir'))
sys.path.append(os.path.join(project_root, 'MODELS', 'LogisticRegression'))

import logistic_regression as lr
import vector_representation as vr

# 1. Cargar datasets
datasets_dir = os.path.join(project_root, 'DATASETS', 'SPLIT')
train = pd.read_csv(os.path.join(datasets_dir, "twitter_trainBALANCED.csv"))

external_csv_path = os.path.join(current_dir, "external_clean_balanced.csv")
if not os.path.exists(external_csv_path):
     external_csv_path = os.path.join(project_root, "external_clean_balanced.csv")

if os.path.exists(external_csv_path):
    df_kaggle = pd.read_csv(external_csv_path)
else:
    print("Warning: external_clean_balanced.csv not found.")
    df_kaggle = pd.DataFrame(columns=['text', 'label'])

# 2. Cargar vectores TF-IDF ya guardados
vectors_path = os.path.join(project_root, 'DATASETS', 'VECTORS', 'tfidf')
X_train, _, _, _ = vr.load_tfidf(prefix=vectors_path)
y_train = train['label']

# 3. Vectorizar el test externo usando el mismo vectorizador
vectorizer_path = os.path.join(project_root, 'DATASETS', 'VECTORS', 'tfidf_vectorizer.pkl')
if os.path.exists(vectorizer_path) and not df_kaggle.empty:
    vectorizer = joblib.load(vectorizer_path)
    X_test = vectorizer.transform(df_kaggle.iloc[:, 0])
    y_test = df_kaggle.iloc[:, 1]
else:
    X_test = []
    y_test = []

# 4. Entrenar el modelo con tu train
HYPERPARAMETERS = {
    'C': 2.0,
    'max_iter': 500,
    'penalty': 'l2',
    'class_weight': 'balanced'
}
model = lr.model_one_vs_rest(X_train, y_train, **HYPERPARAMETERS)


# 5. Predecir sobre el test de Kaggle
if len(X_test) > 0:
    test_pred = model.predict(X_test)
else:
    test_pred = []

from collections import Counter

output_lines = []
neutral_count_before = Counter(test_pred).get('neutral', 0)
total_preds = len(test_pred)
proportion_neutral = neutral_count_before / total_preds if total_preds > 0 else 0
output_lines.append(f"Número de predicciones 'neutral' ANTES de reasignar: {neutral_count_before}")
output_lines.append(f"Proporción de 'neutral' ANTES de reasignar: {proportion_neutral:.4f}")

# 5b. Reasignar "neutral" a la clase con mayor probabilidad entre positive/negative
if hasattr(model, 'predict_proba') and len(X_test) > 0:
    proba = model.predict_proba(X_test)
    classes = model.classes_
    idx_neutral = list(classes).index('neutral') if 'neutral' in classes else None
    idx_positive = list(classes).index('positive') if 'positive' in classes else None
    idx_negative = list(classes).index('negative') if 'negative' in classes else None
    test_pred_adj = []
    for i, pred in enumerate(test_pred):
        if pred == 'neutral' and idx_neutral is not None:
            prob_pos = proba[i, idx_positive] if idx_positive is not None else 0
            prob_neg = proba[i, idx_negative] if idx_negative is not None else 0
            if prob_pos >= prob_neg:
                reassigned = 'positive'
            else:
                reassigned = 'negative'
            test_pred_adj.append(reassigned)
        else:
            test_pred_adj.append(pred)
    test_pred = test_pred_adj

# 6. Analizar resultados
from sklearn.metrics import classification_report, accuracy_score

if len(y_test) > 0 and len(test_pred) > 0:
    output_lines.append(f"Accuracy en test Kaggle: {accuracy_score(y_test, test_pred):.4f}")
    output_lines.append(classification_report(y_test, test_pred))
    output_lines.append("Conteo de clases en test Kaggle:")
    for clase, count in Counter(test_pred).items():
        output_lines.append(f"{clase}: {count}")

with open(os.path.join(current_dir, "output_Kaggle.txt"), "w", encoding="utf-8") as f:
    for line in output_lines:
        f.write(line + "\n")