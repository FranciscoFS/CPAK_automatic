"""Outliers actuales con datos batch corregidos"""
import pandas as pd
from pathlib import Path

BASE = Path(r'C:\Users\franc\OneDrive\Desktop\Proyectos\CPAK\reports\validation_round\Mediciones')
ffs = pd.read_excel(BASE / 'mediciones_FFS.xlsx')
ia = pd.read_excel(BASE / 'mediciones_IA.xlsx')

merged = pd.merge(ffs, ia, on=['RUT', 'ID estudio'], suffixes=('_FFS', '_IA'))

for metric in ['HKA', 'mLDFA', 'mMPTA']:
    print(f"\n{'='*80}")
    print(f"  {metric} — TOP 10 ERRORES ABSOLUTOS")
    print(f"{'='*80}")
    
    rows = []
    for side in ['Der', 'Izq']:
        cf = f'{metric}_{side}_FFS'
        ci = f'{metric}_{side}_IA'
        for _, r in merged.iterrows():
            fv, iv = r[cf], r[ci]
            if pd.notna(fv) and pd.notna(iv):
                rows.append({'RUT': r['RUT'], 'ID': r['ID estudio'], 'Lado': side,
                            'FFS': fv, 'IA': iv, 'diff': fv - iv, 'abs_diff': abs(fv - iv)})
    
    df = pd.DataFrame(rows).sort_values('abs_diff', ascending=False).head(10)
    for _, r in df.iterrows():
        flag = ' ⚠️ SIGNO' if (r['FFS'] > 0) != (r['IA'] > 0) and abs(r['diff']) > 2 else ''
        flag += ' 🔴' if r['abs_diff'] > 5 else ''
        print(f"  {r['RUT']:15s} ID={r['ID']:>8d}  {r['Lado']:4s}  "
              f"FFS={r['FFS']:>7.2f}°  IA={r['IA']:>7.2f}°  "
              f"dif={r['diff']:>+6.2f}°{flag}")
