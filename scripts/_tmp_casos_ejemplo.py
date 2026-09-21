"""Mostrar datos completos de los casos ejemplo"""
import pandas as pd
from pathlib import Path
import json

BASE = Path(r'C:\Users\franc\OneDrive\Desktop\Proyectos\CPAK\reports\validation_round\Mediciones')
ffs = pd.read_excel(BASE / 'mediciones_FFS.xlsx')
ia = pd.read_excel(BASE / 'mediciones_IA.xlsx')

jsons_dir = Path(r'C:\Users\franc\OneDrive\Desktop\Proyectos\CPAK\visualizations\final_pipeline_batch')

casos = [
    ('4133820-2', 5541353, '4133820-2_5541353'),
    ('5849636-7', 6724761, '5849636-7_6724761'),
]

for rut, id_est, pattern in casos:
    print(f"\n{'='*70}")
    print(f"CASO EJEMPLO: RUT={rut}  ID Estudio={id_est}")
    print(f"{'='*70}")
    
    # FFS
    fr = ffs[(ffs['RUT'] == rut) & (ffs['ID estudio'] == id_est)]
    ir = ia[(ia['RUT'] == rut) & (ia['ID estudio'] == id_est)]
    
    if len(fr) == 0 or len(ir) == 0:
        print("  No encontrado en datos")
        continue
    
    fr = fr.iloc[0]
    ir = ir.iloc[0]
    
    for side in ['Der', 'Izq']:
        print(f"\n  {side}:")
        for m in ['HKA', 'mLDFA', 'mMPTA']:
            fv = fr[f'{m}_{side}']
            iv = ir[f'{m}_{side}']
            if pd.notna(fv) and pd.notna(iv):
                diff = fv - iv
                print(f"    {m}:  FFS={fv:>7.2f}°  IA={iv:>7.2f}°  dif={diff:+7.2f}°")
    
    # Buscar JSON
    for jf in jsons_dir.glob(f"*{pattern}*.json"):
        with open(jf) as f:
            data = json.load(f)
        print(f"\n  JSON: {jf.name}")
        print(f"  Image: {data.get('image', 'N/A')}")
        for side in ['Der', 'Izq']:
            m = data.get('sides',{}).get(side,{}).get('metrics',{})
            pts = data.get('sides',{}).get(side,{}).get('points',{})
            print(f"  {side}: HKA={m.get('HKA')}  LDFA={m.get('LDFA')}  MPTA={m.get('MPTA')}  CPAK={m.get('CPAK_type')}")
            if pts:
                print(f"    Keypoints: {len(pts)} puntos")
                for k,v in sorted(pts.items()):
                    print(f"      P{k}: ({v[0]:.0f}, {v[1]:.0f})")
        break
