"""IA vs promedio de humanos (FFS+JS)."""
import pandas as pd, numpy as np
from pathlib import Path
B = Path('reports/validation_round/Mediciones')
ia = pd.read_excel(B/'mediciones_IA.xlsx')
ffs = pd.read_excel(B/'mediciones_FFS.xlsx')
js = pd.read_excel(B/'mediciones_JS.xlsx')

# Merge por RUT + ID estudio
m = ia.merge(ffs, on=['RUT','ID estudio'], suffixes=('_IA','_FFS'))
m = m.merge(js, on=['RUT','ID estudio'])
sides = ['Der','Izq']
metrics = ['HKA','mLDFA','mMPTA']

print(f'Total pares: {len(m)}')
print()
print(f'{"Metric":8s} {"Error sistematico":>20s} {"DE":>8s} {"MAE":>8s} {"LoA inf":>8s} {"LoA sup":>8s} {"CCC":>8s}')
print('-'*70)

for met in metrics:
    diffs = []
    for s in sides:
        kia = f'{met}_{s}_IA'
        kffs = f'{met}_{s}_FFS'
        kjs = f'{met}_{s}'
        if kia in m.columns and kffs in m.columns and kjs in m.columns:
            for _, row in m.iterrows():
                v_ia = row[kia]
                v_ffs = row[kffs]
                v_js = row[kjs]
                if pd.notna(v_ia) and pd.notna(v_ffs) and pd.notna(v_js):
                    prom_humano = (v_ffs + v_js) / 2
                    diffs.append(v_ia - prom_humano)
    
    if diffs:
        d = np.array(diffs)
        bias = np.mean(d)
        sd = np.std(d, ddof=1)
        mae = np.mean(np.abs(d))
        loa_low = bias - 1.96*sd
        loa_up = bias + 1.96*sd
        # CCC
        # Reconstruir arrays para CCC
        ia_vals = np.array([v_ia for _, row in m.iterrows() for s in sides 
                          if pd.notna(row.get(f'{met}_{s}_IA')) 
                          and pd.notna(row.get(f'{met}_{s}_FFS'))
                          and pd.notna(row.get(f'{met}_{s}')) 
                          for v_ia in [row[f'{met}_{s}_IA']] 
                          for v_ffs in [row[f'{met}_{s}_FFS']] 
                          for v_js in [row[f'{met}_{s}']] 
                          if pd.notna(v_ia) and pd.notna(v_ffs) and pd.notna(v_js)])
        # simpler
        ia_list = [row[f'{met}_{s}_IA'] for _, row in m.iterrows() for s in sides 
                   if pd.notna(row[f'{met}_{s}_IA']) and pd.notna(row[f'{met}_{s}_FFS']) and pd.notna(row[f'{met}_{s}'])]
        hu_list = [(row[f'{met}_{s}_FFS'] + row[f'{met}_{s}'])/2 for _, row in m.iterrows() for s in sides 
                   if pd.notna(row[f'{met}_{s}_IA']) and pd.notna(row[f'{met}_{s}_FFS']) and pd.notna(row[f'{met}_{s}'])]
        
        ia_arr = np.array(ia_list)
        hu_arr = np.array(hu_list)
        mx, my = np.mean(hu_arr), np.mean(ia_arr)
        vx, vy = np.var(hu_arr, ddof=0), np.var(ia_arr, ddof=0)
        r = np.corrcoef(hu_arr, ia_arr)[0,1]
        ccc = 2*r*np.sqrt(vx)*np.sqrt(vy) / (vx + vy + (mx-my)**2)
        
        n = len(diffs)
        print(f'{met:8s} {bias:>+8.3f} +/- {sd:.3f}  {mae:>8.3f}  {loa_low:>+8.3f} {loa_up:>+8.3f}  {ccc:>8.4f}  (n={n})')
