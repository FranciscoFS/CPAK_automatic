"""Check available observer data and compute ICC + weighted kappa for all observers."""
import pandas as pd
import numpy as np
from pathlib import Path
import pingouin as pg
from sklearn.metrics import cohen_kappa_score

BASE = Path('reports/validation_round/Mediciones')

# ── Clasificación CPAK ──
def classify_ahka(val):
    if pd.isna(val): return None
    if val < -2.0: return "Varo"
    if val <= 2.0: return "Neutro"
    return "Valgo"

def classify_jlo(val):
    if pd.isna(val): return None
    if val < 177.0: return "Apex Distal"
    if val <= 181.0: return "Neutro"
    return "Apex Proximal"

AHKA_ORDER = {"Varo": -1, "Neutro": 0, "Valgo": 1}
JLO_ORDER = {"Apex Distal": -1, "Neutro": 0, "Apex Proximal": 1}

# ── Cargar todos los observadores ──
obs_files = {
    'FFS': BASE / 'mediciones_FFS.xlsx',
    'IA': BASE / 'mediciones_IA.xlsx',
    'JS': BASE / 'mediciones_JS.xlsx',
}

records = []
for observer, fpath in obs_files.items():
    df = pd.read_excel(fpath)
    for _, row in df.iterrows():
        for side in ['Der', 'Izq']:
            hka = row.get(f'HKA_{side}', np.nan)
            ldfa = row.get(f'mLDFA_{side}', np.nan)
            mpta = row.get(f'mMPTA_{side}', np.nan)
            # Clasificar CPAK desde ángulos manuales o IA
            ahka_val = mpta - ldfa if pd.notna(mpta) and pd.notna(ldfa) else np.nan
            jlo_val = mpta + ldfa if pd.notna(mpta) and pd.notna(ldfa) else np.nan
            records.append({
                'RUT': str(row.get('RUT', '')).strip(),
                'ID_estudio': str(row.get('ID estudio', '')).strip(),
                'Lado': side,
                'Observador': observer,
                'HKA': hka,
                'mLDFA': ldfa,
                'mMPTA': mpta,
                'aHKA': ahka_val,
                'JLO': jlo_val,
                'aHKA_class': classify_ahka(ahka_val),
                'JLO_class': classify_jlo(jlo_val),
            })

df_long = pd.DataFrame(records)
print(f"Total registros: {len(df_long)}")
print(f"Observadores: {df_long['Observador'].value_counts().to_dict()}")

# ── ICC (igual que antes) ──
metrics = ['HKA', 'mLDFA', 'mMPTA']
for metric in metrics:
    data = df_long.dropna(subset=[metric]).copy()
    data['RUT_Lado'] = data['RUT'] + '_' + data['ID_estudio'] + '_' + data['Lado']
    icc_df = pg.intraclass_corr(data=data, targets='RUT_Lado', raters='Observador', ratings=metric, nan_policy='omit')
    icc_row = icc_df[icc_df['Type'] == 'ICC(A,1)']
    if len(icc_row) == 0:
        icc_row = icc_df.iloc[[0]]
    print(f"\n=== {metric} ===")
    print(f"  ICC(A,1)={icc_row['ICC'].values[0]:.4f}  IC95%: [{icc_row['CI95'].values[0][0]:.4f}, {icc_row['CI95'].values[0][1]:.4f}]")

# ── Weighted Kappa para aHKA class y JLO class, y simple kappa para CPAK type ──
print("\n" + "="*70)
print("  KAPPA PONDERADO (Weighted κ) — Clasificaciones CPAK")
print("="*70)

for metric_name, class_col, order_map in [
    ("aHKA class (Varo/Neutro/Valgo)", 'aHKA_class', AHKA_ORDER),
    ("JLO class (Distal/Neutro/Proximal)", 'JLO_class', JLO_ORDER),
]:
    print(f"\n  {metric_name}:")
    for obs1, obs2 in [('IA', 'FFS'), ('IA', 'JS'), ('FFS', 'JS')]:
        sub = df_long[df_long['Observador'].isin([obs1, obs2])].dropna(subset=[class_col]).copy()
        sub['RUT_Lado'] = sub['RUT'] + '_' + sub['ID_estudio'] + '_' + sub['Lado']
        pivot = sub.pivot_table(index='RUT_Lado', columns='Observador', values=class_col, aggfunc='first').dropna()
        if obs1 in pivot.columns and obs2 in pivot.columns:
            vals1 = pivot[obs1].values
            vals2 = pivot[obs2].values
            wk = cohen_kappa_score(vals1, vals2, weights='linear')
            k = cohen_kappa_score(vals1, vals2)
            n = len(vals1)
            print(f"    {obs1:3s} vs {obs2:3s}: κ ponderado={wk:.4f}  κ simple={k:.4f}  (n={n})")

# ── Kappa simple para CPAK type (I-IX) ──
print("\n  CPAK type (I–IX):")
_CPAK_TYPE = {
    ("Varo", "Apex Distal"): "I", ("Neutro", "Apex Distal"): "II", ("Valgo", "Apex Distal"): "III",
    ("Varo", "Neutro"): "IV", ("Neutro", "Neutro"): "V", ("Valgo", "Neutro"): "VI",
    ("Varo", "Apex Proximal"): "VII", ("Neutro", "Apex Proximal"): "VIII", ("Valgo", "Apex Proximal"): "IX",
}
df_long['CPAK_type'] = df_long.apply(
    lambda r: _CPAK_TYPE.get((r['aHKA_class'], r['JLO_class']), None) 
    if r['aHKA_class'] and r['JLO_class'] else None, axis=1)

for obs1, obs2 in [('IA', 'FFS'), ('IA', 'JS'), ('FFS', 'JS')]:
    sub = df_long[df_long['Observador'].isin([obs1, obs2])].dropna(subset=['CPAK_type']).copy()
    sub['RUT_Lado'] = sub['RUT'] + '_' + sub['ID_estudio'] + '_' + sub['Lado']
    pivot = sub.pivot_table(index='RUT_Lado', columns='Observador', values='CPAK_type', aggfunc='first').dropna()
    if obs1 in pivot.columns and obs2 in pivot.columns:
        vals1 = pivot[obs1].values
        vals2 = pivot[obs2].values
        k = cohen_kappa_score(vals1, vals2)
        n = len(vals1)
        agree = sum(vals1[i] == vals2[i] for i in range(n))
        pct = agree / n * 100
        print(f"    {obs1:3s} vs {obs2:3s}: λ={k:.4f}  acuerdo={pct:.1f}%  (n={n})")

# ── Fleiss' Kappa (multi-rater) para los 3 observadores ──
print("\n  FLEISS' KAPPA (multi-rater IA+FFS+JS):")
from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters

for metric_name, class_col, cats in [
    ("aHKA class (Varo/Neutro/Valgo)", 'aHKA_class', ["Varo", "Neutro", "Valgo"]),
    ("JLO class (Distal/Neutro/Proximal)", 'JLO_class', ["Apex Distal", "Neutro", "Apex Proximal"]),
    ("CPAK type (I–IX)", 'CPAK_type', [str(i) for i in range(1,10)]),
]:
    sub = df_long[df_long['Observador'].isin(['IA', 'FFS', 'JS'])].dropna(subset=[class_col]).copy()
    sub['RUT_Lado'] = sub['RUT'] + '_' + sub['ID_estudio'] + '_' + sub['Lado']
    # Pivot: rows = RUT_Lado, cols = Observador
    pivot = sub.pivot_table(index='RUT_Lado', columns='Observador', values=class_col, aggfunc='first').dropna()
    if pivot.shape[1] >= 3:
        mat = pivot[['IA', 'FFS', 'JS']].values
        # Agregar en formato de frecuencias por categoría
        agg, _ = aggregate_raters(mat)
        fk = fleiss_kappa(agg)
        n = len(mat)
        print(f"    {metric_name}: Fleiss κ={fk:.4f}  (n={n})")

# ── CCC (Concordance Correlation Coefficient) fórmula directa ──
print("\n" + "="*70)
print("  CCC (Lin) — FFS como gold standard")
print("="*70)
def ccc_lin(x, y):
    mx, my = np.mean(x), np.mean(y)
    vx, vy = np.var(x, ddof=0), np.var(y, ddof=0)
    r = np.corrcoef(x, y)[0,1]
    return 2*r*np.sqrt(vx)*np.sqrt(vy) / (vx + vy + (mx-my)**2)

for metric in metrics:
    sub = df_long[df_long['Observador'].isin(['IA', 'FFS'])].dropna(subset=[metric]).copy()
    sub['RUT_Lado'] = sub['RUT'] + '_' + sub['ID_estudio'] + '_' + sub['Lado']
    pivot = sub.pivot_table(index='RUT_Lado', columns='Observador', values=metric, aggfunc='first').dropna()
    if 'IA' in pivot.columns and 'FFS' in pivot.columns:
        ia = pivot['IA'].values
        ffs = pivot['FFS'].values
        ccc = ccc_lin(ffs, ia)
        r = np.corrcoef(ffs, ia)[0,1]
        bias = np.mean(ia - ffs)
        print(f"  {metric}: CCC={ccc:.4f}  Pearson r={r:.4f}  Bias (IA−FFS)={bias:.4f}°")

