"""ICC y MAE por grado KL, y ver qué casos se quedan sin datos"""
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(r'C:\Users\franc\OneDrive\Desktop\Proyectos\CPAK\reports\validation_round')
ffs = pd.read_excel(BASE / 'Mediciones' / 'mediciones_FFS.xlsx')
ia = pd.read_excel(BASE / 'Mediciones' / 'mediciones_IA.xlsx')
ann = pd.read_csv(BASE / 'round_annotations.csv')

m = pd.merge(ffs, ia, on=['RUT', 'ID estudio'], suffixes=('_FFS', '_IA'))

# Crear key por RUT+ID
ann['key'] = ann['id_rut_candidate'].astype(str).str.strip() + '_' + ann['id_study_candidate'].astype(str).str.strip()
m['key'] = m['RUT'].astype(str).str.strip() + '_' + m['ID estudio'].astype(str).str.strip()

# Ver cuántos matchean
matched = m['key'].isin(ann['key'])
print(f"Total estudios: {len(m)}")
print(f"Con match en annotations: {matched.sum()}")
print(f"Sin match: {(~matched).sum()}")
print(f"\nEstudios SIN match en annotations:")
for _, r in m[~matched].iterrows():
    print(f"  RUT={r['RUT']}  ID={r['ID estudio']}")

# Ahora análisis por KL grado (solo los que tienen match)
m_ann = m[matched].merge(ann[['key','prosthesis_der','prosthesis_izq','kl_der','kl_izq']], on='key', how='left')

def get_kl(row, side):
    col = f'kl_{side.lower()}'
    v = row.get(col)
    return int(v) if pd.notna(v) else None

print("\n" + "=" * 80)
print("ICC y MAE POR GRADO KL")
print("=" * 80)

for met in ['HKA', 'mLDFA', 'mMPTA']:
    print(f"\n  {met}")
    print(f"  {'KL':>5} {'N':>5} {'ICC':>8} {'Pearson r':>10} {'MAE':>8} {'Bias':>8}")
    print(f"  {'-'*44}")
    
    for kl_grade in [0, 1, 2, 3, 4]:
        ffs_vals, ia_vals = [], []
        for s in ['Der', 'Izq']:
            cf, ci = f'{met}_{s}_FFS', f'{met}_{s}_IA'
            for _, r in m_ann.iterrows():
                kl = get_kl(r, s)
                if kl == kl_grade and pd.notna(r[cf]) and pd.notna(r[ci]):
                    # Excluir prótesis del análisis de KL
                    prot_col = f'prosthesis_{s.lower()}'
                    if r.get(prot_col) == True:
                        continue
                    ffs_vals.append(r[cf])
                    ia_vals.append(r[ia])
        
        n = len(ffs_vals)
        if n < 3:
            print(f"  {kl_grade:>5} {n:>5}  → insuficiente")
            continue
        
        ffs_a = np.array(ffs_vals)
        ia_a = np.array(ia_vals)
        diff = ffs_a - ia_a
        r_pearson = np.corrcoef(ffs_a, ia_a)[0, 1] if np.std(ffs_a) > 0 and np.std(ia_a) > 0 else np.nan
        
        means = (ffs_a + ia_a) / 2
        gm = np.mean(means)
        msb = np.sum((means - gm)**2) * 2 / (n - 1)
        msw = np.sum((ffs_a - means)**2 + (ia_a - means)**2) / n
        icc = (msb - msw) / (msb + msw) if msb > 0 else 0
        
        mae = np.mean(np.abs(diff))
        bias = np.mean(diff)
        print(f"  {kl_grade:>5} {n:>5} {icc:>8.3f} {r_pearson:>10.3f} {mae:>8.2f} {bias:>+8.2f}")
    
    # Prótesis aparte
    prot_ffs, prot_ia = [], []
    for s in ['Der', 'Izq']:
        cf, ci = f'{met}_{s}_FFS', f'{met}_{s}_IA'
        for _, r in m_ann.iterrows():
            prot_col = f'prosthesis_{s.lower()}'
            if r.get(prot_col) == True and pd.notna(r[cf]) and pd.notna(r[ci]):
                prot_ffs.append(r[cf])
                prot_ia.append(r[ci])
    
    n = len(prot_ffs)
    if n >= 3:
        ffs_a = np.array(prot_ffs)
        ia_a = np.array(prot_ia)
        diff = ffs_a - ia_a
        r_pearson = np.corrcoef(ffs_a, ia_a)[0, 1]
        means = (ffs_a + ia_a) / 2
        gm = np.mean(means)
        msb = np.sum((means - gm)**2) * 2 / (n - 1)
        msw = np.sum((ffs_a - means)**2 + (ia_a - means)**2) / n
        icc = (msb - msw) / (msb + msw) if msb > 0 else 0
        mae = np.mean(np.abs(diff))
        bias = np.mean(diff)
        print(f"  {'Prót':>5} {n:>5} {icc:>8.3f} {r_pearson:>10.3f} {mae:>8.2f} {bias:>+8.2f}")
