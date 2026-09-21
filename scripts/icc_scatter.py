"""
Cálculo de ICC y scatter plots: FFS (manual) vs IA (automático).
Métricas: HKA, mLDFA, mMPTA — lado Der e Izq como observaciones independientes.

Uso: conda run -n physis_seg python scripts/icc_scatter.py
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 11, 'font.family': 'DejaVu Sans'})
from pathlib import Path

BASE = Path(r'C:\Users\franc\OneDrive\Desktop\Proyectos\CPAK\reports\validation_round\Mediciones')
OUT_DIR = Path(r'C:\Users\franc\OneDrive\Desktop\Proyectos\CPAK\visualizations_abstract')

# ── 1. Cargar datos en formato long ────────────────────────────────────
obs_files = {'FFS': BASE / 'mediciones_FFS.xlsx', 'IA': BASE / 'mediciones_IA.xlsx'}

records = []
for observer, fpath in obs_files.items():
    df = pd.read_excel(fpath)
    df = df.dropna(subset=['HKA_Der', 'HKA_Izq'], how='all').copy()
    for _, row in df.iterrows():
        for side in ['Der', 'Izq']:
            hka = row.get(f'HKA_{side}', np.nan)
            ldfa = row.get(f'mLDFA_{side}', np.nan)
            mpta = row.get(f'mMPTA_{side}', np.nan)
            if pd.notna(hka) or pd.notna(ldfa) or pd.notna(mpta):
                records.append({
                    'RUT': row['RUT'],
                    'ID_estudio': row.get('ID estudio', ''),
                    'Lado': side,
                    'Observador': observer,
                    'HKA': hka,
                    'mLDFA': ldfa,
                    'mMPTA': mpta,
                })

df_long = pd.DataFrame(records)

# ── 2. Armar pares FFS-IA por (RUT, Lado) para cada métrica ──────────
metrics = ['HKA', 'mLDFA', 'mMPTA']
metric_labels = {
    'HKA': 'HKA (°)',
    'mLDFA': 'mLDFA (°)',
    'mMPTA': 'mMPTA (°)',
}

results = {}

for metric in metrics:
    pivot = df_long.pivot_table(index=['RUT', 'ID_estudio', 'Lado'], columns='Observador', values=metric).dropna()
    ffs_vals = pivot['FFS'].values
    ia_vals = pivot['IA'].values
    n_pairs = len(ffs_vals)
    
    # ICC(2,1) con pingouin
    import pingouin as pg
    # Crear target compuesto RUT_Lado para tratar cada (paciente, lado) como sujeto único
    icc_data = df_long.dropna(subset=[metric]).copy()
    icc_data['RUT_Lado'] = icc_data['RUT'] + '_' + icc_data['ID_estudio'].astype(str) + '_' + icc_data['Lado']
    icc_df = pg.intraclass_corr(data=icc_data, targets='RUT_Lado', raters='Observador', ratings=metric, nan_policy='omit')
    # ICC(A,1) = two-way random effects, single measure, absolute agreement (equivalente a ICC2,1)
    icc_row = icc_df[icc_df['Type'] == 'ICC(A,1)']
    if len(icc_row) == 0:
        icc_row = icc_df.iloc[[0]]
    icc_val = icc_row['ICC'].values[0]
    ci95 = icc_row['CI95'].values[0]
    icc_ci_low = ci95[0]
    icc_ci_high = ci95[1]
    
    # Bland-Altman
    diff = ffs_vals - ia_vals
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    loa_upper = mean_diff + 1.96 * std_diff
    loa_lower = mean_diff - 1.96 * std_diff
    
    # Correlación Pearson
    pearson_r = np.corrcoef(ffs_vals, ia_vals)[0, 1]
    
    # MAE, RMSE
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff**2))
    
    results[metric] = {
        'n': n_pairs,
        'ICC': icc_val,
        'ICC_CI_low': icc_ci_low,
        'ICC_CI_high': icc_ci_high,
        'Pearson_r': pearson_r,
        'MAE': mae,
        'RMSE': rmse,
        'Bias (FFS-IA)': mean_diff,
        'LoA lower': loa_lower,
        'LoA upper': loa_upper,
        'ffs': ffs_vals,
        'ia': ia_vals,
    }

# ── 3. Imprimir tabla de resultados ────────────────────────────────────
print("=" * 100)
print(f"{'Métrica':<10} {'N':>5} {'ICC':>8} {'IC95% inf':>10} {'IC95% sup':>10} {'Pearson r':>10} {'MAE':>8} {'RMSE':>8} {'Bias':>8} {'LoA inf':>8} {'LoA sup':>8}")
print("-" * 100)
for metric in metrics:
    r = results[metric]
    print(f"{metric:<10} {r['n']:>5} {r['ICC']:>8.4f} {r['ICC_CI_low']:>10.4f} {r['ICC_CI_high']:>10.4f} {r['Pearson_r']:>10.4f} {r['MAE']:>8.3f} {r['RMSE']:>8.3f} {r['Bias (FFS-IA)']:>8.3f} {r['LoA lower']:>8.3f} {r['LoA upper']:>8.3f}")
print("=" * 100)

# ── 4. Scatter plots + Bland-Altman ───────────────────────────────────
OUT_DIR.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(3, 2, figsize=(14, 16))
fig.suptitle('FFS (manual) vs IA (automático) — CPAK Validation', fontsize=15, fontweight='bold', y=0.99)

colors = {'HKA': '#2196F3', 'mLDFA': '#FF5722', 'mMPTA': '#4CAF50'}

for i, metric in enumerate(metrics):
    r = results[metric]
    ffs = r['ffs']
    ia = r['ia']
    color = colors[metric]
    
    # ── Scatter plot (izquierda) ──
    ax = axes[i, 0]
    ax.scatter(ia, ffs, alpha=0.6, s=40, color=color, edgecolors='white', linewidth=0.5)
    
    # Línea identidad y=x
    lims = [min(ffs.min(), ia.min()) - 2, max(ffs.max(), ia.max()) + 2]
    ax.plot(lims, lims, 'k--', alpha=0.4, linewidth=1, label='Identidad')
    
    # Línea de regresión
    m, b = np.polyfit(ia, ffs, 1)
    x_fit = np.linspace(lims[0], lims[1], 100)
    ax.plot(x_fit, m * x_fit + b, color=color, alpha=0.5, linewidth=1.5, linestyle='-', label=f'Regresión (R²={r["Pearson_r"]**2:.3f})')
    
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel(f'IA — {metric_labels[metric]}', fontsize=11)
    ax.set_ylabel(f'FFS — {metric_labels[metric]}', fontsize=11)
    ax.set_title(f'{metric_labels[metric]}   |   ICC={r["ICC"]:.3f}  MAE={r["MAE"]:.2f}°  N={r["n"]}', fontsize=12)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.15)
    ax.legend(fontsize=8, loc='lower right')
    
    # Text box con métricas
    textstr = f'N = {r["n"]}\nICC = {r["ICC"]:.3f} ({r["ICC_CI_low"]:.3f}–{r["ICC_CI_high"]:.3f})\nPearson r = {r["Pearson_r"]:.3f}\nMAE = {r["MAE"]:.2f}°\nRMSE = {r["RMSE"]:.2f}°'
    props = dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8, edgecolor=color)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=8.5,
            verticalalignment='top', bbox=props)
    
    # ── Bland-Altman (derecha) ──
    ax2 = axes[i, 1]
    mean_vals = (ffs + ia) / 2
    diff = ffs - ia
    
    ax2.scatter(mean_vals, diff, alpha=0.6, s=40, color=color, edgecolors='white', linewidth=0.5)
    ax2.axhline(y=r['Bias (FFS-IA)'], color='red', linestyle='-', linewidth=1.5, label=f"Bias = {r['Bias (FFS-IA)']:.2f}°")
    ax2.axhline(y=r['LoA upper'], color='red', linestyle='--', linewidth=1, alpha=0.6)
    ax2.axhline(y=r['LoA lower'], color='red', linestyle='--', linewidth=1, alpha=0.6)
    ax2.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    ax2.fill_between([mean_vals.min() - 2, mean_vals.max() + 2],
                     r['LoA lower'], r['LoA upper'],
                     alpha=0.08, color='red')
    
    ax2.set_xlabel('Media (FFS + IA) / 2 (°)', fontsize=11)
    ax2.set_ylabel('Diferencia FFS − IA (°)', fontsize=11)
    ax2.set_title(f'Bland-Altman — {metric_labels[metric]}', fontsize=12)
    ax2.grid(True, alpha=0.15)
    ax2.legend(fontsize=8, loc='upper right')
    
    # Etiquetas LoA
    ax2.text(mean_vals.max() - 1, r['LoA upper'] + 0.3, f"+1.96 SD = {r['LoA upper']:.2f}°",
             fontsize=7.5, color='red', ha='right', va='bottom')
    ax2.text(mean_vals.max() - 1, r['LoA lower'] - 0.3, f"-1.96 SD = {r['LoA lower']:.2f}°",
             fontsize=7.5, color='red', ha='right', va='top')

plt.tight_layout(rect=[0, 0, 1, 0.97])
scatter_path = OUT_DIR / '02_icc_scatter_blandaltman.png'
fig.savefig(scatter_path, dpi=200, bbox_inches='tight')
print(f"\nScatter + Bland-Altman guardado: {scatter_path}")
plt.close()

print("\n✅ Listo.")
