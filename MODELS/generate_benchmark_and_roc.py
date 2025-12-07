import os
import sys
import joblib
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

try:
    import plotly.graph_objects as go
except ImportError:
    print("Error: Plotly library is required for interactive plots. Please install it using 'pip install plotly'.")
    sys.exit(1)

# ==========================================
# SETUP PATHS & IMPORTS
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))

# Change working directory to project root
os.chdir(project_root)

# Add project root to sys.path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from AnalizarLimpiarDividir.vector_representation import load_and_vectorize_splits
except ImportError:
    print("Error: Could not import 'load_and_vectorize_splits'. Check your directory structure.")
    sys.exit(1)

# ==========================================
# TASK 2: GENERATE AND PLOT ROC CURVES
# ==========================================
def generate_roc_curves():
    print("--- Generating ROC Curves (Validation Set - Macro Average) ---")
    
    # Find all .joblib files in MODELS
    model_files = []
    for root, dirs, files in os.walk(os.path.join(project_root, 'MODELS')):
        for file in files:
            if file.endswith('.joblib'):
                model_files.append(os.path.join(root, file))

    if not model_files:
        print("No .joblib models found in MODELS directory.")
        return

    # Initialize Plotly Figure
    fig = go.Figure()
    fig.add_shape(
        type='line', line=dict(dash='dash', color='navy'),
        x0=0, x1=1, y0=0, y1=1
    )

    models_plotted = 0
    ranking_data = []

    # Cache loaded data to avoid reloading for same config
    data_cache = {}

    for model_path in model_files:
        rel_path = os.path.relpath(model_path, project_root)
        
        # Skip vectorizer cache files (they are in DATASETS/VECTORS usually, but just in case)
        if 'tfidf' in os.path.basename(model_path).lower() and 'joblib' in os.path.basename(model_path).lower():
             # Heuristic: if it looks like a vectorizer cache (e.g. tfidf_500.joblib), skip it
             # But wait, some models might be named 'logistic_tfidf.joblib'.
             # Vectorizer cache is usually in DATASETS/VECTORS. We are scanning MODELS.
             pass

        try:
            print(f"Loading model: {rel_path} ...")
            model = joblib.load(model_path)

            # Determine method (BOW vs TFIDF)
            filename = os.path.basename(model_path).lower()
            if 'bow' in filename:
                method = 'BOW'
            else:
                method = 'TFIDF'

            # Determine max_features
            if hasattr(model, 'n_features_in_'):
                n_features = model.n_features_in_
            else:
                # Fallback or skip
                # Some pipelines might wrap the vectorizer, but here we assume model expects vectorized input
                print(f"  [Skipping] Could not determine 'n_features_in_' for {rel_path}.")
                continue

            # Load data
            cache_key = (method, n_features)
            if cache_key in data_cache:
                data = data_cache[cache_key]
            else:
                print(f"  Loading data (method={method}, max_features={n_features})...")
                data = load_and_vectorize_splits(method=method, max_features=n_features)
                data_cache[cache_key] = data

            # Use Validation set instead of Test set
            if 'X_val' in data and 'y_val' in data:
                X_eval = data['X_val']
                y_eval = data['y_val']
                set_name = "Validation"
            else:
                print("  [Warning] 'X_val' not found, falling back to 'X_test'")
                X_eval = data['X_test']
                y_eval = data['y_test']
                set_name = "Test"

            # Predict probabilities or decision function
            y_score = None
            if hasattr(model, "predict_proba"):
                try:
                    y_score = model.predict_proba(X_eval)
                except Exception as e:
                    print(f"  [Info] predict_proba failed: {e}. Trying decision_function...")
            
            if y_score is None and hasattr(model, "decision_function"):
                try:
                    y_score = model.decision_function(X_eval)
                    print(f"  [Info] Using decision_function for {rel_path}.")
                except Exception as e:
                    print(f"  [Skipping] decision_function failed: {e}")
                    continue
            
            if y_score is None:
                print(f"  [Skipping] Model does not support predict_proba or decision_function.")
                continue

            # Get classes
            if hasattr(model, 'classes_'):
                classes = model.classes_
            else:
                classes = np.unique(y_eval)
                print(f"  [Info] Model has no classes_ attribute. Using np.unique(y_eval): {classes}")

            # Binarize labels for ROC
            y_eval_bin = label_binarize(y_eval, classes=classes)
            n_classes = y_eval_bin.shape[1]

            # Handle binary case vs multiclass shapes
            if n_classes == 1 and y_score.shape[1] == 2:
                 # Binary classification fix: label_binarize returns 1 col, predict_proba returns 2
                 n_classes = 2
                 y_eval_bin = np.hstack((1 - y_eval_bin, y_eval_bin))

            # Compute ROC curve and ROC area for each class
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            
            valid_score_shape = True
            if y_score.shape[1] != n_classes:
                 print(f"  [Skipping] Shape mismatch: y_eval_bin {y_eval_bin.shape}, y_score {y_score.shape}")
                 valid_score_shape = False
            
            if not valid_score_shape:
                continue

            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_eval_bin[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            # Compute Macro-Average ROC curve and ROC area
            # First aggregate all false positive rates
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

            # Then interpolate all ROC curves at this points
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(n_classes):
                mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

            # Finally average it and compute AUC
            mean_tpr /= n_classes

            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr
            roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

            # Plot Macro-Average
            model_name = os.path.basename(rel_path).replace('.joblib', '')
            clean_name = model_name.replace('_', ' ').title()
            
            fig.add_trace(go.Scatter(
                x=fpr["macro"], 
                y=tpr["macro"],
                mode='lines',
                name=f'{clean_name} (AUC = {roc_auc["macro"]:.2f})',
                hovertemplate=f'<b>{clean_name}</b><br>FPR: %{{x:.3f}}<br>TPR: %{{y:.3f}}<br>AUC: {roc_auc["macro"]:.2f}<extra></extra>'
            ))

            models_plotted += 1
            print(f"  Successfully processed {rel_path} (Macro AUC={roc_auc['macro']:.2f})")
            
            ranking_data.append((clean_name, roc_auc["macro"]))

        except Exception as e:
            print(f"  [Error] Failed to process {rel_path}: {e}")
            continue

    if models_plotted > 0:
        # Print Ranking
        print("\n" + "="*40)
        print("RANKING OF MODELS BY MACRO-AVERAGE AUC")
        print("="*40)
        ranking_data.sort(key=lambda x: x[1], reverse=True)
        for i, (name, score) in enumerate(ranking_data, 1):
            print(f"{i}. {name}: {score:.4f}")
        print("="*40 + "\n")

        fig.update_layout(
            title='Macro-Average ROC Curve Comparison (Validation Set)',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1.05]),
            width=1000, height=800,
            legend=dict(x=0.6, y=0.05)
        )

        output_dir = os.path.join(project_root, 'GRAFIQUES', 'BENCHMARK')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'all_models_roc.html')

        fig.write_html(output_path)
        print(f"\nInteractive ROC plot saved to: {output_path}")
    else:
        print("\nNo models were successfully processed. No plot generated.")

if __name__ == "__main__":
    generate_roc_curves()
