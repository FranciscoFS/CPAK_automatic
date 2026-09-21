"""Minimalist scatter + Bland-Altman: IA vs avg of observers (FFS+JS)."""
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path('reports/validation_round/Mediciones')
OUT_DIR = Path('visualizations_abstract')
ia = pd.read_excel(BASE/'mediciones_IA.xlsx')
ffs = pd.read_excel(BASE/'mediciones_FFS.xlsx')
js = pd.read_excel(BASE/'mediciones_JS.xlsx')
m = ia.merge(ffs, on=['RUT','ID estudio'], suffixes=('_IA','_FFS')).merge(js, on=['RUT','ID estudio'])

sides = ['Der','Izq']
metrics = ['HKA', 'mLDFA', 'mMPTA']
labels = {'HKA': 'HKA (°)', 'mLDFA': 'mLDFA (°)', 'mMPTA': 'mMPTA (°)'}
colors = {'HKA': '#2196F3', 'mLDFA': '#E65100', 'mMPTA': '#2E7D32'}

data = {}
# ── ICC multi-observador correcto (IA+FFS+JS, 3 raters) ──
import pingouin as pg
all_long = []
for met in metrics:
    records = []
    for _, row in m.iterrows():
        for s in sides:
            kia = f'{met}_{s}_IA'; kffs = f'{met}_{s}_FFS'; kjs = f'{met}_{s}'
            via, vffs, vjs = row[kia], row[kffs], row[kjs]
            if pd.notna(via) and pd.notna(vffs) and pd.notna(vjs):
                rid = f"{row['RUT']}_{row['ID estudio']}_{s}"
                records.append({'id': rid, 'IA': via, 'FFS': vffs, 'JS': vjs})
    df = pd.DataFrame(records)
    df_long = df.melt(id_vars='id', var_name='Obs', value_name=met)
    icc_df = pg.intraclass_corr(data=df_long, targets='id', raters='Obs', ratings=met)
    icc_row = icc_df[icc_df['Type'] == 'ICC(A,1)']
    icc_multi = icc_row['ICC'].values[0]
    
    # Datos para scatter y BA: IA vs promedio observadores
    ia_vals, hu_vals = df['IA'].values, (df['FFS'].values + df['JS'].values) / 2
    diff = ia_vals - hu_vals
    mean = (ia_vals + hu_vals) / 2
    bias = np.mean(diff)
    sd = np.std(diff, ddof=1)
    r = np.corrcoef(hu_vals, ia_vals)[0, 1]
    mae = np.mean(np.abs(diff))
    
    data[met] = {
        'ia': ia_vals, 'hu': hu_vals, 'diff': diff, 'mean': mean,
        'bias': bias, 'sd': sd, 'loa_low': bias - 1.96*sd, 'loa_up': bias + 1.96*sd,
        'icc': icc_multi, 'r': r, 'mae': mae, 'n': len(ia_vals)
    }

# ── Figure ──
fig, axes = plt.subplots(3, 2, figsize=(7.5, 10))
fig.subplots_adjust(left=0.1, right=0.95, bottom=0.05, top=0.96, hspace=0.35, wspace=0.35)

for i, met in enumerate(metrics):
    d = data[met]
    color = colors[met]
    
    # ── Scatter (col 0) ──
    ax = axes[i, 0]
    ax.scatter(d['hu'], d['ia'], alpha=0.55, s=28, color=color, edgecolors='white', linewidth=0.3, zorder=3)
    lims = [min(d['hu'].min(), d['ia'].min()) - 1, max(d['hu'].max(), d['ia'].max()) + 1]
    ax.plot(lims, lims, 'k-', alpha=0.25, linewidth=0.8, zorder=1)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_aspect('equal')
    ax.tick_params(labelsize=7)
    ax.set_xlabel("Observers' average (°)", fontsize=8)
    ax.set_ylabel('AI (°)', fontsize=8)
    
    # Anotación compacta
    txt = f'ICC = {d["icc"]:.3f}\nMAE = {d["mae"]:.2f}°\nn = {d["n"]}'
    ax.text(0.05, 0.95, txt, transform=ax.transAxes, fontsize=7,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, linewidth=0))
    
    ax.set_title(f'{met}', fontsize=9, fontweight='bold', pad=2)
    ax.grid(True, alpha=0.15)
    
    # ── Bland-Altman (col 1) ──
    ax2 = axes[i, 1]
    ax2.scatter(d['mean'], d['diff'], alpha=0.55, s=28, color=color, edgecolors='white', linewidth=0.3, zorder=3)
    ax2.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    ax2.axhline(y=d['bias'], color='red', linestyle='-', linewidth=0.8, label=f'{d["bias"]:.2f}°')
    ax2.axhline(y=d['loa_up'], color='red', linestyle='--', linewidth=0.6, alpha=0.6)
    ax2.axhline(y=d['loa_low'], color='red', linestyle='--', linewidth=0.6, alpha=0.6)
    ax2.fill_between([d['mean'].min()-1, d['mean'].max()+1], d['loa_low'], d['loa_up'],
                     alpha=0.06, color='red')
    ax2.tick_params(labelsize=7)
    ax2.set_xlabel('Mean (°)', fontsize=8)
    ax2.set_ylabel('AI − Observers (°)', fontsize=8)
    ax2.grid(True, alpha=0.15)
    ax2.legend(fontsize=6, loc='upper right', framealpha=0.8)

# Column headers in English
axes[0, 0].set_title('HKA — Scatter', fontsize=10, fontweight='bold')
axes[0, 1].set_title('HKA — Bland-Altman', fontsize=10, fontweight='bold')

fig.savefig(OUT_DIR / 'abstract_ia_vs_humanos.png', dpi=300, bbox_inches='tight')
print(f"Guardado: {OUT_DIR / 'abstract_ia_vs_humanos.png'}")

# Copy as Figure 2 in MANUSCRITO
import shutil
fig2_path = Path('MANUSCRITO/Figure 2.png')
shutil.copyfile(OUT_DIR / 'abstract_ia_vs_humanos.png', fig2_path)
print(f"Copiado: {fig2_path}")
plt.close()
print("✅ Listo.")
