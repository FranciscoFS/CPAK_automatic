"""Test heteroscedasticity in Bland-Altman: correlation |diff| vs mean."""
import pandas as pd, numpy as np
from scipy import stats
from pathlib import Path
B = Path('reports/validation_round/Mediciones')
ia = pd.read_excel(B/'mediciones_IA.xlsx')
ffs = pd.read_excel(B/'mediciones_FFS.xlsx')
js = pd.read_excel(B/'mediciones_JS.xlsx')
merged = ia.merge(ffs, on=['RUT','ID estudio'], suffixes=('_IA','_FFS')).merge(js, on=['RUT','ID estudio'])

sides = ['Der','Izq']
metrics = ['HKA', 'mLDFA', 'mMPTA']

for met in metrics:
    ia_vals, hu_vals = [], []
    for _, row in merged.iterrows():
        for s in sides:
            kia = f'{met}_{s}_IA'
            kffs = f'{met}_{s}_FFS'
            kjs = f'{met}_{s}'
            v_ia, v_ffs, v_js = row[kia], row[kffs], row[kjs]
            if pd.notna(v_ia) and pd.notna(v_ffs) and pd.notna(v_js):
                ia_vals.append(v_ia)
                hu_vals.append((v_ffs + v_js) / 2)
    
    diff = np.array(ia_vals) - np.array(hu_vals)
    mean_vals = (np.array(ia_vals) + np.array(hu_vals)) / 2
    abs_diff = np.abs(diff)
    
    # Pearson correlación |diff| vs mean
    r, p = stats.pearsonr(mean_vals, abs_diff)
    # Spearman (más robusto)
    rs, ps = stats.spearmanr(mean_vals, abs_diff)
    # Breusch-Pagan más simple: regresión lineal |diff| ~ mean
    slope, intercept, r2, p_reg, se = stats.linregress(mean_vals, abs_diff)
    
    print(f'\n=== {met} ===')
    print(f'  Pearson r(|diff|, media) = {r:.4f}  (p={p:.4f})')
    print(f'  Spearman ρ(|diff|, media) = {rs:.4f}  (p={ps:.4f})')
    print(f'  Pendiente regresión: {slope:.4f}  (R²={r2:.4f}, p={p_reg:.4f})')
    if p_reg < 0.05:
        print(f'  ⚠ HETEROSCEDASTICIDAD DETECTADA (p={p_reg:.4f})')
    else:
        print(f'  ✓ Homocedástico (p={p_reg:.4f})')
    
    # Rango de medias para ver si hay zonas con más error
    bins = np.percentile(mean_vals, [0, 25, 50, 75, 100])
    print(f'  |diff| por cuartil de media:')
    for i in range(4):
        mask = (mean_vals >= bins[i]) & (mean_vals < bins[i+1])
        print(f'    {bins[i]:.1f}–{bins[i+1]:.1f}°: media |diff|={abs_diff[mask].mean():.3f}° (n={mask.sum()})')
