"""Bias (systematic error) for all observer pairs."""
import pandas as pd, numpy as np
from pathlib import Path
B = Path('reports/validation_round/Mediciones')
obs = {o: pd.read_excel(B/f'mediciones_{o}.xlsx') for o in ['IA','FFS','JS']}
sides = ['Der','Izq']; metrics = ['HKA','mLDFA','mMPTA']
pairs = [('IA','FFS'),('IA','JS'),('FFS','JS')]
for n1, n2 in pairs:
    d1, d2 = obs[n1], obs[n2]
    print(f'=== {n1} vs {n2} ===')
    for m in metrics:
        diffs = []
        for s in sides:
            k = f'{m}_{s}'
            if k in d1.columns and k in d2.columns:
                merged = d1[['RUT','ID estudio']].merge(d2[['RUT','ID estudio',k]], on=['RUT','ID estudio'])
                merged = merged.merge(d1[['RUT','ID estudio',k]], on=['RUT','ID estudio'], suffixes=('_2','_1'))
                diffs.extend((merged[f'{k}_1'] - merged[f'{k}_2']).dropna().tolist())
        if diffs:
            b = np.mean(diffs); sd = np.std(diffs, ddof=1); n = len(diffs)
            loa_low = b - 1.96*sd; loa_up = b + 1.96*sd
            print(f'  {m}: error sistematico = {b:.3f} +/- {sd:.3f}  (n={n})')
            print(f'      LoA: {loa_low:.3f} a {loa_up:.3f}')
