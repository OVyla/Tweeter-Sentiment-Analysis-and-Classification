import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar resultados
results_df = pd.read_csv('./RandomSearch/svm_random_search_results.csv')
results_clean = results_df.dropna(subset=['mean_test_score'])

print("\n" + "="*80)
print("GENERANDO GRÁFICA: RANKING DE IMPORTANCIA DE PARÁMETROS")
print("="*80)

param_info = [
    ('param_C', 'C (Regularización)'),
    ('param_loss', 'Loss (Función de Pérdida)'),
    ('param_dual', 'Dual (Formulación)'),
    ('param_tol', 'Tol (Tolerancia)'),
    ('param_max_iter', 'Max Iter (Iteraciones)'),
    ('param_fit_intercept', 'Fit Intercept'),
    ('param_intercept_scaling', 'Intercept Scaling'),
    ('param_multi_class', 'Multi Class (Estrategia)'),
]

# Calcular impacto
impact_data = []
for param_name, param_label in param_info:
    param_analysis = results_clean.groupby(param_name)['mean_test_score'].agg(['mean', 'std'])
    max_acc = param_analysis['mean'].max()
    min_acc = param_analysis['mean'].min()
    impact = (max_acc - min_acc) * 100
    impact_data.append({
        'Parámetro': param_label,
        'Impacto (%)': impact,
        'Min Acc': min_acc,
        'Max Acc': max_acc
    })

impact_df = pd.DataFrame(impact_data).sort_values('Impacto (%)', ascending=True)

# 4. Ranking de importancia - SOLO GRÁFICA (MÁS GRANDE)
fig, ax = plt.subplots(figsize=(18, 10))

impact_sorted = impact_df.sort_values('Impacto (%)', ascending=False)

# Gráfica: Ranking
colors_rank = plt.cm.RdYlGn(np.linspace(0.3, 0.8, len(impact_sorted)))
bars = ax.bar(range(len(impact_sorted)), impact_sorted['Impacto (%)'], 
             color=colors_rank, edgecolor='black', linewidth=3, alpha=0.85, width=0.65)
ax.set_xticks(range(len(impact_sorted)))
ax.set_xticklabels(impact_sorted['Parámetro'], rotation=45, ha='right', fontsize=14, fontweight='bold')
ax.set_ylabel('Impacto en Accuracy (%)', fontsize=16, fontweight='bold')
ax.set_title('Ranking de Importancia de Parámetros SVM', fontsize=18, fontweight='bold', pad=25)
ax.grid(True, alpha=0.3, axis='y', linewidth=1.5)

for i, (bar, val) in enumerate(zip(bars, impact_sorted['Impacto (%)'])):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.25, f'#{i+1}\n{val:.2f}%',
            ha='center', va='bottom', fontweight='bold', fontsize=13)

ax.set_ylim(0, max(impact_sorted['Impacto (%)']) + 1.5)
ax.tick_params(axis='y', labelsize=12)

plt.tight_layout()
plt.savefig('RANKING_IMPORTANCIA.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n" + "="*80)
print("✓ ¡GRÁFICA GENERADA!")
print("="*80)
print("\nGráfica creada:")
print("  RANKING_IMPORTANCIA.png - Ranking de importancia de parámetros SVM")
print("\n✓ Solo la gráfica del ranking!")
print("="*80 + "\n")
