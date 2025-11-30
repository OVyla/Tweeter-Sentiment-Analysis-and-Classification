import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import sys
import os

# Añadir la ruta al vector_representation y random_forest_model si es necesario
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from vector_representation import load_tfidf
from RandomForest.random_forest_model import rf_standard, rf_one_vs_rest, ada_boost, extra_trees, gradient_boosting, ada_boost_ovr, lightgbm_ovr, lightgbm_multiclass

# Cargar datos usando rutas absolutas basadas en la ubicación de este script
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
train = pd.read_csv(os.path.join(base_dir, 'twitter_trainBALANCED.csv'))
val = pd.read_csv(os.path.join(base_dir, 'twitter_valBALANCED.csv'))
test = pd.read_csv(os.path.join(base_dir, 'twitter_testBALANCED.csv'))



# Cargar datos vectorizados (TF-IDF) previamente guardados
X_train, X_val, X_test, _ = load_tfidf(prefix="./VECTORES/tfidf")

y_train = train['label']
y_val = val['label']
y_test = test['label']

OUTPUT_FILE = "output_random_forest.txt"

def save_report(f, model_name, title, y_true, y_pred):
    f.write(f"\n=== {model_name} ===\n")
    f.write(f"--- {title} ---\n")
    f.write(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}\n")
    f.write(f"Confusion Matrix:\n{confusion_matrix(y_true, y_pred)}\n")
    f.write(classification_report(y_true, y_pred))
    f.write("-" * 60 + "\n")

models_to_test = [
    # (
    #     "Random Forest OVR (gini)",
    #     lambda: rf_one_vs_rest(
    #         X_train, y_train,
    #         n_estimators=200,
    #         max_depth=30,
    #         min_samples_leaf=10,
    #         min_samples_split=20,
    #         max_features='sqrt',
    #         n_jobs=-1,
    #         criterion='gini'
    #     )
    # ),
    # (
    #     "AdaBoost OVR",
    #     lambda: ada_boost_ovr(
    #         X_train, y_train,
    #         n_estimators=70,
    #         random_state=42
    #     )
    # ),
    # (
    #     "ExtraTrees OVR",
    #     lambda: extra_trees_ovr(
    #         X_train, y_train,
    #         n_estimators=100,
    #         random_state=42,
    #         n_jobs=-1
    #     )
    # ),
    # (
    #     "Gradient Boosting OVR",
    #     lambda: gradient_boosting_ovr(
    #         X_train, y_train,
    #         random_state=42
    #     )
    # ),
    # (
    #     "Random Forest OVR (entropy)",
    #     lambda: rf_one_vs_rest(
    #         X_train, y_train,
    #         n_estimators=200,
    #         max_depth=30,
    #         min_samples_leaf=10,
    #         min_samples_split=20,
    #         max_features='sqrt',
    #         n_jobs=-1,
    #         criterion='entropy'
    #     )
    # ),
    # (
    #     "Random Forest OVO (ajustado, n_estimators=30, max_depth=25, min_samples_leaf=5, min_samples_split=5)",
    #     lambda: rf_one_vs_one(
    #         X_train, y_train,
    #         n_estimators=30,
    #         max_depth=25,
    #         min_samples_leaf=5,
    #         min_samples_split=5
    #     )
    # ),
    # (
    #     "LightGBM OVR (tuned)",
    #     lambda: lightgbm_ovr(
    #         X_train, y_train,
    #         n_estimators=600,
    #         learning_rate=0.05,
    #         num_leaves=45,
    #         bagging_fraction=0.8,
    #         feature_fraction=0.8,
    #         lambda_l1=1.0,
    #         lambda_l2=1.0,
    #         min_data_in_leaf=20,
    #         random_state=42,
    #         n_jobs=-1
    #     )
    # ),
    (
        "LightGBM Multiclass (improved)",
        lambda: lightgbm_multiclass(
            X_train, y_train,
            objective="multiclass",
            num_class=3,
            n_estimators=1000,
            learning_rate=0.03,
            num_leaves=70,
            bagging_fraction=0.8,
            feature_fraction=0.8,
            lambda_l1=1.0,
            lambda_l2=1.0,
            min_data_in_leaf=20,
            bagging_freq=1,
            random_state=42,
            n_jobs=-1
        )
    ),
]

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("RESUMEN RANDOM FOREST - Comparativa\n")
    f.write("=" * 40 + "\n")
    for model_name, model_fn in models_to_test:
        f.write(f"\n>>> {model_name} <<<\n")
        model = model_fn()
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)
        save_report(f, model_name, "TRAIN", y_train, train_pred)
        save_report(f, model_name, "VALIDATION", y_val, val_pred)
        save_report(f, model_name, "TEST", y_test, test_pred)

print(f"Resultados guardados en {OUTPUT_FILE}")
