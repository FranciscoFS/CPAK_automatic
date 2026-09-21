"""Gráfico CPAK: distribución FFS (manual) vs IA (automático) en grilla 3×3"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

BASE = Path(r'C:\Users\franc\OneDrive\Desktop\Proyectos\CPAK\reports\validation_round\Mediciones')
OUT = Path(r'C:\Users\franc\OneDrive\Desktop\Proyectos\CPAK\MANUSCRITO\ejemplos')

df = pd.read_excel(BASE / 'cpak_classification.xlsx')

# Separar FFS e IA
ffs = df[df['Fuente']=='Manual'].copy()
ia = df[df['Fuente']=='IA'].copy()

# ── Figura 1: Scatter CPAK (aHKA vs JLO) con grilla ──
fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
fig.suptitle('Distribución CPAK — FFS (manual) vs IA (automático)', 
             fontsize=14, fontweight='bold', y=1.02)

colors_ffs = {'I':'#2196F3','II':'#4CAF50','III':'#FF9800',
              'IV':'#00BCD4','V':'#9C27B0','VI':'#FF5722',
              'VII':'#607D8B','VIII':'#795548','IX':'#F44336'}
colors_ia = {'I':'#90CAF9','II':'#A5D6A7','III':'#FFCC80',
             'IV':'#80DEEA','V':'#CE93D8','VI':'#FFAB91',
             'VII':'#B0BEC5','VIII':'#A1887F','IX':'#EF9A9A'}

for idx, (label, data, cols, marker, size) in enumerate([
    ('FFS (manual)', ffs, colors_ffs, 'o', 80),
    ('IA (automático)', ia, colors_ia, 's', 60)
]):
    ax = axes[idx]
    
    # Grilla CPAK 3×3
    for i in range(4):
        ax.axhline(y=177 + i*2, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    for i in range(4):
        ax.axvline(x=-4 + i*4, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Zonas CPAK
    cpak_labels = {
        (0,0):'I', (0,1):'IV', (0,2):'VII',
        (1,0):'II', (1,1):'V', (1,2):'VIII',
        (2,0):'III', (2,1):'VI', (2,2):'IX'
    }
    for (ci, cj), ctype in cpak_labels.items():
        x_center = -2 + ci * 4
        y_center = 178 + cj * 2
        ax.text(x_center, y_center, ctype, fontsize=18, fontweight='bold',
                color='lightgray', ha='center', va='center', alpha=0.5)
    
    # Límites de zona
    ax.axvline(x=-2, color='black', linewidth=1.5, alpha=0.4)
    ax.axvline(x=2, color='black', linewidth=1.5, alpha=0.4)
    ax.axhline(y=177, color='black', linewidth=1.5, alpha=0.4)
    ax.axhline(y=181, color='black', linewidth=1.5, alpha=0.4)
    
    # Scatter
    for _, r in data.iterrows():
        cpak = r['CPAK']
        c = cols.get(cpak, 'gray')
        ax.scatter(r['aHKA'], r['JLO'], c=c, s=size, marker=marker,
                   edgecolors='white', linewidth=0.5, alpha=0.7, zorder=5)
    
    ax.set_xlabel('aHKA = MPTA − LDFA (°)', fontsize=11)
    ax.set_ylabel('JLO = MPTA + LDFA (°)', fontsize=11)
    ax.set_title(label, fontsize=12, fontweight='bold')
    ax.set_xlim(-10, 18)
    ax.set_ylim(165, 188)
    ax.grid(True, alpha=0.1)
    ax.set_aspect(1.8)

plt.tight_layout()
out_path = OUT.parent / 'cpak_distribucion_comparada.png'
fig.savefig(out_path, dpi=200, bbox_inches='tight')
print(f"Scatter CPAK guardado: {out_path}")

# ── Figura 2: Barras comparativas ──
fig2, ax2 = plt.subplots(figsize=(10, 5))
cpak_types = ['I','II','III','IV','V','VI','VII','VIII','IX']
ffs_counts = [len(ffs[ffs['CPAK']==c]) for c in cpak_types]
ia_counts = [len(ia[ia['CPAK']==c]) for c in cpak_types]

x = np.arange(len(cpak_types))
w = 0.35
bars1 = ax2.bar(x - w/2, ffs_counts, w, label='FFS (manual)', color='#2196F3', edgecolor='white')
bars2 = ax2.bar(x + w/2, ia_counts, w, label='IA (automático)', color='#FF5722', edgecolor='white')

for bar in bars1:
    h = bar.get_height()
    if h > 0:
        ax2.text(bar.get_x() + bar.get_width()/2, h + 0.3, str(int(h)), ha='center', fontsize=9, fontweight='bold')
for bar in bars2:
    h = bar.get_height()
    if h > 0:
        ax2.text(bar.get_x() + bar.get_width()/2, h + 0.3, str(int(h)), ha='center', fontsize=9, fontweight='bold')

ax2.set_xlabel('Tipo CPAK', fontsize=12)
ax2.set_ylabel('N° de rodillas (lados)', fontsize=12)
ax2.set_title('Distribución de tipos CPAK: FFS vs IA', fontsize=13, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(cpak_types, fontsize=11)
ax2.legend(fontsize=11)
ax2.grid(axis='y', alpha=0.2)

plt.tight_layout()
out_path2 = OUT.parent / 'cpak_barras_comparadas.png'
fig2.savefig(out_path2, dpi=200, bbox_inches='tight')
print(f"Barras CPAK guardado: {out_path2}")
plt.close('all')
print("✅ Listo.")
