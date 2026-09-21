"""Revisar estado actual de TODOS los Excel de mediciones (post-recopilación completa)"""
import pandas as pd
from pathlib import Path

BASE = Path(r'C:\Users\franc\OneDrive\Desktop\Proyectos\CPAK\reports\validation_round\Mediciones')
files = ['mediciones_FFS.xlsx','mediciones_IA.xlsx','mediciones_JS.xlsx','mediciones_PB.xlsx']

for f in files:
    print(f'\n{"="*60}')
    print(f'=== {f} ===')
    print('='*60)
    df = pd.read_excel(BASE / f)
    cols = [c for c in df.columns if c not in ('RUT','ID estudio','tiempo')]
    has_data = df.dropna(subset=cols, how='all')
    print(f'Total filas: {df.shape[0]}')
    print(f'Filas con datos: {has_data.shape[0]}')
    print(f'RUTs únicos con datos: {has_data["RUT"].nunique()}')
    print(f'Columnas: {list(df.columns)}')
    if has_data.shape[0] > 0:
        print(has_data.head(3).to_string())
