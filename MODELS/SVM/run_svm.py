import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import sys
import os

# Añadir la ruta al vector_representation y svm_model si es necesario
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from vector_representation import load_tfidf
from SVM.svm_model import svm_standard, svm_one_vs_one, svm_one_vs_rest
from SVM.svm_model import svm_standard, svm_one_vs_one, svm_one_vs_rest


# Cargar datos usando rutas absolutas basadas en la ubicación de este script
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
train = pd.read_csv(os.path.join(base_dir, 'twitter_trainBALANCED.csv'))
val = pd.read_csv(os.path.join(base_dir, 'twitter_valBALANCED.csv'))
test = pd.read_csv(os.path.join(base_dir, 'twitter_testBALANCED.csv'))



# Cargar datos vectorizados TF-IDF extendidos (con longitud)
import joblib
X_train = joblib.load("./VECTORES/tfidf_X_train.pkl")
X_val = joblib.load("./VECTORES/tfidf_X_val.pkl")
X_test = joblib.load("./VECTORES/tfidf_X_test.pkl")

y_train = train['label']
y_val = val['label']
y_test = test['label']


OUTPUT_FILE = "output_svm.txt"

def save_report(f, model_name, title, y_true, y_pred):
	f.write(f"\n=== {model_name} ===\n")
	f.write(f"--- {title} ---\n")
	f.write(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}\n")
	f.write(f"Confusion Matrix:\n{confusion_matrix(y_true, y_pred)}\n")
	f.write(classification_report(y_true, y_pred))
	f.write("-" * 60 + "\n")

models_to_test = [
	# Descomenta solo el modelo que quieras probar:
	("Linear SVM (OVR)", lambda: svm_one_vs_rest(X_train, y_train, C=0.1)),
	#("Linear SVM (OVO)", lambda: svm_one_vs_one(X_train, y_train, C=0.1)),
	
]

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
	f.write("RESUMEN SVM - Comparativa\n")
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
