import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# CARGAR RESULTADOS DEL RANDOM SEARCH
# ==========================================
results_df = pd.read_csv("lr_random_search_results.csv")
results_df_valid = results_df.dropna(subset=['mean_test_score'])

print(f"Total de iteraciones válidas: {len(results_df_valid)}")

# ==========================================
# CALCULAR IMPORTANCIA CORRECTA
# ==========================================
# Importancia = desviación estándar de los scores para cada valor de parámetro
# Parámetros con mayor varianza en scores = más importantes

param_importance = {}

params_to_analyze = {
    'param_C': 'C',
    'param_penalty': 'penalty',
    'param_solver': 'solver',
    'param_max_iter': 'max_iter',
    'param_fit_intercept': 'fit_intercept',
    'param_intercept_scaling': 'intercept_scaling',
    'param_class_weight': 'class_weight',
    'param_multi_class': 'multi_class'
}

for param_col, param_name in params_to_analyze.items():
    try:
        # Agrupar por parámetro y calcular variancia de scores
        param_groups = results_df_valid.groupby(param_col)['mean_test_score'].agg(['mean', 'std', 'count']).reset_index()
        
        # Importancia = promedio de std (qué tan variable es el score según este parámetro)
        # + diferencia entre max y min scores
        max_score = param_groups['mean'].max()
        min_score = param_groups['mean'].min()
        range_score = max_score - min_score
        avg_std = param_groups['std'].mean()
        
        # Combinar rango y variabilidad
        importance = range_score * 100 + avg_std * 100
        param_importance[param_name] = {
            'range': range_score,
            'avg_std': avg_std,
            'total_importance': importance
        }
    except:
        pass

# Ordenar por importancia total
param_importance_sorted = dict(sorted(param_importance.items(), 
                                     key=lambda x: x[1]['total_importance'], 
                                     reverse=True))

print("\n" + "="*70)
print("RANKING DE IMPORTANCIA DE PARÁMETROS (CORREGIDO)")
print("="*70)
print(f"{'Rank':<6} {'Parámetro':<20} {'Rango':<10} {'Varianza':<10} {'Total':<10}")
print("-"*70)
for idx, (param, metrics) in enumerate(param_importance_sorted.items(), 1):
    print(f"{idx:<6} {param:<20} {metrics['range']:>9.4f} {metrics['avg_std']:>9.4f} {metrics['total_importance']:>9.4f}")
print("="*70)

# ==========================================
# GRÁFICA: RANKING DE IMPORTANCIA
# ==========================================
fig, ax = plt.subplots(figsize=(14, 8))

# Orden MANUAL según rango (mayor a menor)
manual_order = ['C', 'intercept_scaling', 'penalty', 'max_iter', 'solver', 'fit_intercept', 'multi_class', 'class_weight']
manual_values = [0.3927, 0.1846, 0.1686, 0.1491, 0.1337, 0.0767, 0.0394, 0.0000]

x_pos = np.arange(len(manual_order))
colors = plt.cm.RdYlGn(np.linspace(0.8, 0.2, len(manual_order)))

# Gráfica de barras horizontales
bars1 = ax.barh(x_pos, manual_values, color=colors, alpha=0.85, edgecolor='black', linewidth=1.5)

ax.set_yticks(x_pos)
ax.set_yticklabels(manual_order, fontsize=12, fontweight='bold')
ax.set_xlabel('Importancia (Rango de Scores)', fontsize=13, fontweight='bold')
ax.set_title('Logistic Regression - Ranking de Importancia de Parámetros', fontsize=15, fontweight='bold')
ax.grid(axis='x', alpha=0.3, linestyle='--')

# Añadir valores
for bar, value in zip(bars1, manual_values):
    width = bar.get_width()
    ax.text(width, bar.get_y() + bar.get_height()/2.,
            f' {value:.4f}',
            ha='left', va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('05_lr_importance_ranking.png', dpi=300, bbox_inches='tight')
print("\n✓ Guardada: 05_lr_importance_ranking.png")
plt.close()

# ==========================================
# TOP 5 PARÁMETROS
# ==========================================
print("\n" + "="*70)
print("TOP 5 PARÁMETROS MÁS IMPORTANTES")
print("="*70)
for idx, (param, metrics) in enumerate(list(param_importance_sorted.items())[:5], 1):
    print(f"{idx}. {param:.<15} → Rango: {metrics['range']:.4f} (Score range)")
print("="*70)
