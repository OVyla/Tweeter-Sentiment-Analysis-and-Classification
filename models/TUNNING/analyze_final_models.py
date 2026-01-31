import os
import sys
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# ==========================================
# CONFIGURACIÓ
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.insert(0, project_root)
os.chdir(project_root)

models_dir = os.path.join(current_dir, 'best_models_optuna')
plots_dir = os.path.join(current_dir, 'analysis_plots')
os.makedirs(plots_dir, exist_ok=True)

# ==========================================
# CÀRREGA DE DADES (TEST SET)
# ==========================================
try:
    from preprocessing.vector_representation import load_and_vectorize_splits
    print("Loading Test Data...")
    # Carreguem les dades (necessitem X_test i y_test)
    data = load_and_vectorize_splits(method='TFIDF', max_features=160000)
    X_test = data['X_test']
    y_test = data['y_test']
except ImportError:
    sys.exit("Error: No s'ha pogut importar load_and_vectorize_splits")

# ==========================================
# ANÀLISI DE models
# ==========================================
model_files = [f for f in os.listdir(models_dir) if f.endswith('_best_full.joblib')]
results = []

print(f"\n🔍 Found {len(model_files)} models to analyze.\n")

for file in model_files:
    model_path = os.path.join(models_dir, file)
    model_name = file.replace('_best_full.joblib', '')
    
    print(f"Evaluating: {model_name}...")
    
    try:
        model = joblib.load(model_path)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        results.append({
            'Model': model_name,
            'Accuracy': acc,
            'F1-Score (Weighted)': f1
        })
        
        # --- Matriu de Confusió ---
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'confusion_matrix_{model_name}.png'))
        plt.close()
        
    except Exception as e:
        print(f"❌ Error evaluating {model_name}: {e}")

# ==========================================
# COMPARATIVA GRÀFICA
# ==========================================
if results:
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by='Accuracy', ascending=False)
    
    print("\n📊 Results Summary:")
    print(df_results)
    
    # Guardar resultats en CSV
    df_results.to_csv(os.path.join(plots_dir, 'model_comparison_results.csv'), index=False)

    # Gràfic de Barres Comparatiu
    plt.figure(figsize=(12, 6))
    
    # Melt per tenir Accuracy i F1 en la mateixa llegenda
    df_melted = df_results.melt(id_vars="Model", var_name="Metric", value_name="Score")
    
    sns.barplot(data=df_melted, x="Model", y="Score", hue="Metric", palette="viridis")
    plt.title("Model Comparison: Accuracy vs F1-Score")
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    save_path = os.path.join(plots_dir, 'model_comparison_bar_chart.png')
    plt.savefig(save_path)
    print(f"\n📈 Comparison plot saved to: {save_path}")
    print(f"📂 All plots saved in: {plots_dir}")
else:
    print("No results to plot.")
