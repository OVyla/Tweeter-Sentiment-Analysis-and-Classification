import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score


# --- Cargar splits ---
train = pd.read_csv("DATASETMASDATOS/twitter_trainBALANCED.csv")
val = pd.read_csv("DATASETMASDATOS/twitter_valBALANCED.csv")
test = pd.read_csv("DATASETMASDATOS/twitter_testBALANCED.csv")

# --- Vectorización TF-IDF mejorada ---
vectorizer = TfidfVectorizer(max_features=80000, ngram_range=(1,3), sublinear_tf=True,min_df=2,max_df=0.95 )
X_train = vectorizer.fit_transform(train['text'])
X_val = vectorizer.transform(val['text'])
X_test = vectorizer.transform(test['text'])

y_train = train['label']
y_val = val['label']
y_test = test['label']

# --- Entrenamiento y evaluación directa (rápido) ---
logreg = LogisticRegression(C=2, solver='liblinear', class_weight='balanced', max_iter=300, random_state=42)
logreg.fit(X_train, y_train)

# --- Evaluación en train ---
train_pred = logreg.predict(X_train)
print("\n--- TRAIN ---")
print(classification_report(y_train, train_pred))
print("Accuracy:", accuracy_score(y_train, train_pred))

# --- Validación ---
val_pred = logreg.predict(X_val)
print("\n--- VALIDATION ---")
print(classification_report(y_val, val_pred))
print("Accuracy:", accuracy_score(y_val, val_pred))

# --- Test final ---
test_pred = logreg.predict(X_test)
print("\n--- TEST ---")
print(classification_report(y_test, test_pred))
print("Accuracy:", accuracy_score(y_test, test_pred))





""""
# --- Grid SearchCV más exhaustivo ---
from sklearn.model_selection import GridSearchCV
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

param_grid = {
	'C': [1, 2],
	'solver': ['liblinear', 'saga'],
	'class_weight': ['balanced', None],
	'penalty': ['l1', 'l2'],
	'max_iter': [300]
}

# 'saga' soporta l1 y l2, 'liblinear' solo l1 y l2
@ignore_warnings(category=ConvergenceWarning)
def run_grid_search():
	logreg = LogisticRegression(random_state=42, multi_class='auto')
	gs = GridSearchCV(logreg, param_grid, cv=3, scoring='f1_macro', verbose=2, n_jobs=-1, error_score='raise')
	gs.fit(X_train, y_train)
	return gs

print("\n--- GridSearchCV: buscando mejores hiperparámetros... ---")
gs = run_grid_search()
print("\nMejores hiperparámetros:")
print(gs.best_params_)

# Evaluar el mejor modelo en validación y test
best_logreg = gs.best_estimator_
val_pred_gs = best_logreg.predict(X_val)
print("\n--- VALIDATION (GridSearchCV best) ---")
print(classification_report(y_val, val_pred_gs))
print("Accuracy:", accuracy_score(y_val, val_pred_gs))

test_pred_gs = best_logreg.predict(X_test)
print("\n--- TEST (GridSearchCV best) ---")
print(classification_report(y_test, test_pred_gs))
print("Accuracy:", accuracy_score(y_test, test_pred_gs))
"""