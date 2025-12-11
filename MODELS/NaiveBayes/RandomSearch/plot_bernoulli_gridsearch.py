import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 10

# Cargar resultados
results_df = pd.read_csv("bernoulli_random_search_results.csv")

# Crear figura con 4 subplots
fig = plt.figure(figsize=(18, 12))

# ===== SUBPLOT 1: CV Scores a lo largo de las iteraciones =====
ax1 = plt.subplot(2, 3, 1)
iterations = range(1, len(results_df) + 1)
scores = results_df['mean_test_score'].values

ax1.plot(iterations, scores, 'o-', color='#2E86AB', linewidth=2, markersize=6, alpha=0.7)
ax1.axhline(y=results_df['mean_test_score'].max(), color='#A23B72', linestyle='--', linewidth=2, label=f'Best: {results_df["mean_test_score"].max():.4f}')
ax1.axhline(y=results_df['mean_test_score'].mean(), color='#F18F01', linestyle='--', linewidth=2, label=f'Mean: {results_df["mean_test_score"].mean():.4f}')
ax1.set_xlabel('Iteración', fontsize=11, fontweight='bold')
ax1.set_ylabel('CV Score (Accuracy)', fontsize=11, fontweight='bold')
ax1.set_title('CV Scores a lo largo de iteraciones', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0.3, 0.75])

# ===== SUBPLOT 2: Tiempo de ejecución vs Score =====
ax2 = plt.subplot(2, 3, 2)
fit_times = results_df['mean_fit_time'].values
scores = results_df['mean_test_score'].values

scatter = ax2.scatter(fit_times, scores, c=scores, cmap='viridis', s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
ax2.set_xlabel('Tiempo de ajuste (segundos)', fontsize=11, fontweight='bold')
ax2.set_ylabel('CV Score (Accuracy)', fontsize=11, fontweight='bold')
ax2.set_title('Tiempo vs Rendimiento', fontsize=12, fontweight='bold')
plt.colorbar(scatter, ax=ax2, label='CV Score')
ax2.grid(True, alpha=0.3)

# ===== SUBPLOT 3: Distribución de parámetro Alpha =====
ax3 = plt.subplot(2, 3, 3)
alpha_values = results_df['param_alpha'].values
alpha_scores = results_df['mean_test_score'].values

# Crear scatter con color por score
scatter = ax3.scatter(np.log10(alpha_values), alpha_scores, c=alpha_scores, cmap='plasma', s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
ax3.set_xlabel('Log10(Alpha)', fontsize=11, fontweight='bold')
ax3.set_ylabel('CV Score', fontsize=11, fontweight='bold')
ax3.set_title('Parámetro Alpha vs Rendimiento', fontsize=12, fontweight='bold')
plt.colorbar(scatter, ax=ax3, label='CV Score')
ax3.grid(True, alpha=0.3)

# ===== SUBPLOT 4: Distribución de parámetro Binarize =====
ax4 = plt.subplot(2, 3, 4)
binarize_values = results_df['param_binarize'].values
binarize_scores = results_df['mean_test_score'].values

scatter = ax4.scatter(binarize_values, binarize_scores, c=binarize_scores, cmap='coolwarm', s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
ax4.set_xlabel('Binarize Threshold', fontsize=11, fontweight='bold')
ax4.set_ylabel('CV Score', fontsize=11, fontweight='bold')
ax4.set_title('Parámetro Binarize vs Rendimiento', fontsize=12, fontweight='bold')
plt.colorbar(scatter, ax=ax4, label='CV Score')
ax4.grid(True, alpha=0.3)

# ===== SUBPLOT 5: Distribución de fit_prior =====
ax5 = plt.subplot(2, 3, 5)
fit_prior_true = results_df[results_df['param_fit_prior'] == True]['mean_test_score']
fit_prior_false = results_df[results_df['param_fit_prior'] == False]['mean_test_score']

bp = ax5.boxplot([fit_prior_true, fit_prior_false], labels=['True', 'False'], patch_artist=True)
bp['boxes'][0].set_facecolor('#FF6B6B')
bp['boxes'][1].set_facecolor('#4ECDC4')

ax5.set_ylabel('CV Score', fontsize=11, fontweight='bold')
ax5.set_xlabel('fit_prior', fontsize=11, fontweight='bold')
ax5.set_title('Parámetro fit_prior vs Rendimiento', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3, axis='y')

# Añadir valores
for i, (label, data) in enumerate([(True, fit_prior_true), (False, fit_prior_false)], 1):
    ax5.text(i, data.max() + 0.01, f'Max: {data.max():.4f}', ha='center', fontsize=9)
    ax5.text(i, data.min() - 0.02, f'Min: {data.min():.4f}', ha='center', fontsize=9)

# ===== SUBPLOT 6: Top 10 mejores configuraciones =====
ax6 = plt.subplot(2, 3, 6)
top_10 = results_df.nlargest(10, 'mean_test_score')[['params', 'mean_test_score']].reset_index(drop=True)

colors_gradient = plt.cm.Greens(np.linspace(0.4, 0.9, len(top_10)))
bars = ax6.barh(range(len(top_10)), top_10['mean_test_score'].values, color=colors_gradient, edgecolor='black', linewidth=1.5)

ax6.set_yticks(range(len(top_10)))
ax6.set_yticklabels([f"#{i+1}" for i in range(len(top_10))])
ax6.set_xlabel('CV Score', fontsize=11, fontweight='bold')
ax6.set_title('Top 10 Mejores Configuraciones', fontsize=12, fontweight='bold')
ax6.invert_yaxis()
ax6.grid(True, alpha=0.3, axis='x')

# Añadir valores en las barras
for i, (idx, row) in enumerate(top_10.iterrows()):
    ax6.text(row['mean_test_score'] + 0.003, i, f"{row['mean_test_score']:.4f}", 
             va='center', fontsize=9, fontweight='bold')

plt.suptitle('BERNOULLI NAIVE BAYES - RandomizedSearchCV Analysis\n(50 iteraciones, 3-fold CV)', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('bernoulli_gridsearch_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Gráfica guardada: bernoulli_gridsearch_analysis.png")
plt.close()

# ===== SEGUNDA FIGURA: Análisis por parámetros individuales =====
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Alpha vs Score
ax = axes[0]
sorted_data = results_df.sort_values('param_alpha')
ax.scatter(np.log10(sorted_data['param_alpha']), sorted_data['mean_test_score'], 
          c=sorted_data['mean_test_score'], cmap='RdYlGn', s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
ax.set_xlabel('Log10(Alpha)', fontsize=12, fontweight='bold')
ax.set_ylabel('CV Score', fontsize=12, fontweight='bold')
ax.set_title('Alpha: Impacto en el Rendimiento', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

# Binarize vs Score
ax = axes[1]
sorted_data = results_df.sort_values('param_binarize')
ax.scatter(sorted_data['param_binarize'], sorted_data['mean_test_score'], 
          c=sorted_data['mean_test_score'], cmap='RdYlGn', s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
ax.set_xlabel('Binarize', fontsize=12, fontweight='bold')
ax.set_ylabel('CV Score', fontsize=12, fontweight='bold')
ax.set_title('Binarize: Impacto en el Rendimiento', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

# fit_prior distribution
ax = axes[2]
fit_prior_stats = results_df.groupby('param_fit_prior')['mean_test_score'].agg(['mean', 'std', 'count'])
x_pos = [0, 1]
means = fit_prior_stats['mean'].values
stds = fit_prior_stats['std'].values

bars = ax.bar(x_pos, means, yerr=stds, capsize=10, color=['#FF6B6B', '#4ECDC4'], 
             edgecolor='black', linewidth=2, alpha=0.7)
ax.set_xticks(x_pos)
ax.set_xticklabels(['True', 'False'])
ax.set_ylabel('CV Score (promedio)', fontsize=12, fontweight='bold')
ax.set_title('fit_prior: Impacto en el Rendimiento', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Añadir valores
for i, (mean, std) in enumerate(zip(means, stds)):
    ax.text(i, mean + std + 0.01, f'{mean:.4f}', ha='center', fontsize=11, fontweight='bold')
    ax.text(i, mean - std - 0.02, f'(n={int(fit_prior_stats["count"].values[i])})', ha='center', fontsize=9)

plt.suptitle('Análisis Individual de Parámetros', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('bernoulli_parameters_individual.png', dpi=300, bbox_inches='tight')
print("✓ Gráfica guardada: bernoulli_parameters_individual.png")
plt.close()

# ===== TERCERA FIGURA: Matriz de correlación y heatmap =====
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Heatmap de Score por Alpha y Binarize
ax = axes[0]
pivot_data = results_df.pivot_table(values='mean_test_score', 
                                    index=pd.cut(results_df['param_alpha'], bins=10),
                                    columns=pd.cut(results_df['param_binarize'], bins=10),
                                    aggfunc='mean')
sns.heatmap(pivot_data, cmap='RdYlGn', ax=ax, cbar_kws={'label': 'CV Score'}, vmin=0.33, vmax=0.72)
ax.set_title('Heatmap: Alpha vs Binarize (Score)', fontsize=13, fontweight='bold')
ax.set_xlabel('Binarize (bins)', fontsize=11)
ax.set_ylabel('Alpha (bins)', fontsize=11)

# Distribución de mejores resultados
ax = axes[1]
top_scores = results_df.nlargest(15, 'mean_test_score')

ax.scatter(np.log10(top_scores['param_alpha']), top_scores['param_binarize'], 
          s=300, c=top_scores['mean_test_score'], cmap='Greens', alpha=0.8, 
          edgecolors='black', linewidth=2)

ax.set_xlabel('Log10(Alpha)', fontsize=11, fontweight='bold')
ax.set_ylabel('Binarize', fontsize=11, fontweight='bold')
ax.set_title('Top 15 Configuraciones: Alpha vs Binarize', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

# Etiquetar los puntos superiores
for idx, row in top_scores.head(5).iterrows():
    ax.annotate(f"#{results_df[results_df['mean_test_score'] == row['mean_test_score']].index[0]+1}", 
               xy=(np.log10(row['param_alpha']), row['param_binarize']),
               xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))

plt.suptitle('Análisis de Combinaciones de Parámetros', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('bernoulli_parameters_combined.png', dpi=300, bbox_inches='tight')
print("✓ Gráfica guardada: bernoulli_parameters_combined.png")
plt.close()

# ===== CUARTA FIGURA: Ranking de importancia de parámetros =====
fig, ax = plt.subplots(figsize=(9, 5))

# Calcular importancia relativa de cada parámetro
param_importance = {}

# Alpha: variación en score según cambios de alpha
alpha_sorted = results_df.sort_values('param_alpha')
alpha_importance = alpha_sorted['mean_test_score'].std()

# Binarize: variación en score según cambios de binarize
binarize_sorted = results_df.sort_values('param_binarize')
binarize_importance = binarize_sorted['mean_test_score'].std()

# fit_prior: variación en score según fit_prior
fit_prior_importance = results_df.groupby('param_fit_prior')['mean_test_score'].mean().std()

# Normalizar
importances = {
    'Alpha': alpha_importance,
    'Binarize': binarize_importance,
    'fit_prior': fit_prior_importance
}

total = sum(importances.values())
importances_normalized = {k: v/total for k, v in importances.items()}

# Ordenar por importancia
sorted_imp = dict(sorted(importances_normalized.items(), key=lambda x: x[1], reverse=True))

colors = ['#2E86AB', '#A23B72', '#F18F01']
bars = ax.barh(list(sorted_imp.keys()), list(sorted_imp.values()), color=colors, edgecolor='black', linewidth=2, height=0.5)

ax.set_xlabel('Importancia Relativa (normalizada)', fontsize=12, fontweight='bold')
ax.set_title('Ranking de Importancia de Parámetros\n(Bernoulli Naive Bayes)', fontsize=14, fontweight='bold')
ax.set_xlim([0, max(sorted_imp.values()) * 1.2])

# Aumentar tamaño de los labels del eje Y
ax.tick_params(axis='y', labelsize=13)

# Añadir valores en las barras
for i, (param, importance) in enumerate(sorted_imp.items()):
    ax.text(importance + 0.02, i, f'{importance:.4f} ({importance*100:.1f}%)', 
           va='center', fontsize=11, fontweight='bold')

ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('bernoulli_importance_ranking.png', dpi=300, bbox_inches='tight')
print("✓ Gráfica guardada: bernoulli_importance_ranking.png")
plt.close()

print("\n" + "="*80)
print("RESUMEN DE VISUALIZACIONES GENERADAS")
print("="*80)
print("\n✓ bernoulli_gridsearch_analysis.png - 6 gráficas de análisis general")
print("✓ bernoulli_parameters_individual.png - Análisis individual de cada parámetro")
print("✓ bernoulli_parameters_combined.png - Heatmap y combinaciones")
print("✓ bernoulli_importance_ranking.png - Ranking de importancia")
print("\n" + "="*80 + "\n")
