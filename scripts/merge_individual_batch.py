"""
Fusiona lo mejor de ambos mundos:
- LDFA, MPTA desde individual (mejor detección de keypoints)
- HKA con signo corregido: si aHKA (MPTA-LDFA) < 0 → HKA negativo
"""
import json
from pathlib import Path
import pandas as pd

med = pd.read_excel("reports/validation_round/mediciones.xlsx")
jsons_indiv = Path("visualizations/final_pipeline")
jsons_batch = Path("visualizations/final_pipeline_batch")

# Cargar todos los JSONs individuales
indiv_data = {}
for jf in jsons_indiv.glob("*.json"):
    with open(jf, encoding="utf-8") as f:
        indiv_data[jf.stem] = json.load(f)

batch_data = {}
for jf in jsons_batch.glob("*.json"):
    with open(jf, encoding="utf-8") as f:
        batch_data[jf.stem] = json.load(f)

ok = 0
for i, row in med.iterrows():
    rut = str(row["RUT"]).strip()
    id_est = str(row["ID estudio"]).strip()
    
    # Buscar JSON individual que coincida
    found_indiv = None
    for jname, data in indiv_data.items():
        img = data.get("image", "")
        if id_est in img or (rut in img and id_est not in [x.split('_')[2] for x in indiv_data.keys() if rut in x]):
            # Más específico: ID estudio primero
            pass
        if id_est in img:
            found_indiv = data
            break
    if found_indiv is None:
        for jname, data in indiv_data.items():
            img = data.get("image", "")
            # Fallback: solo RUT si no hay JSONs con ID estudio para este RUT
            has_id_json = any(id_est in jn2 for jn2 in indiv_data.keys() if rut in jn2)
            if rut in img and not has_id_json:
                found_indiv = data
                break
    
    if found_indiv is None:
        continue
    
    for lado_key, col_hka, col_ldfa, col_mpta in [
        ("Der", "HKA_Der", "mLDFA_Der", "mMPTA_Der"),
        ("Izq", "HKA_Izq", "mLDFA_Izq", "mMPTA_Izq"),
    ]:
        lado = found_indiv.get("sides", {}).get(lado_key, {})
        metrics = lado.get("metrics", {})
        if metrics.get("status") != "ok":
            continue
        
        # Tomar LDFA y MPTA del individual (mejores keypoints)
        ldfa = metrics.get("LDFA")
        mpta = metrics.get("MPTA")
        hka_raw = metrics.get("HKA")  # individual, sin signo (siempre positivo)
        
        if ldfa is not None and mpta is not None and hka_raw is not None:
            # Corregir signo de HKA según aHKA = MPTA - LDFA
            ahka = mpta - ldfa
            hka_corregido = hka_raw if ahka >= 0 else -hka_raw
            
            med.at[i, col_hka] = round(hka_corregido, 2)
            med.at[i, col_ldfa] = round(ldfa, 2)
            med.at[i, col_mpta] = round(mpta, 2)
    ok += 1

med.to_excel("reports/validation_round/mediciones.xlsx", index=False)

# Guardar también como mediciones_IA.xlsx
ia_cols = ['RUT', 'ID estudio', 'HKA_Der', 'mLDFA_Der', 'mMPTA_Der',
           'HKA_Izq', 'mLDFA_Izq', 'mMPTA_Izq']
ia_df = med[ia_cols].copy()
ia_df.to_excel("reports/validation_round/Mediciones/mediciones_IA.xlsx", index=False)

print(f"Procesados: {ok} estudios")
print(f"HKA Der con datos: {med['HKA_Der'].notna().sum()}")
print(f"HKA Izq con datos: {med['HKA_Izq'].notna().sum()}")

# Verificar que ahora hay negativos
pos = (med['HKA_Der'] > 0).sum() + (med['HKA_Izq'] > 0).sum()
neg = (med['HKA_Der'] < 0).sum() + (med['HKA_Izq'] < 0).sum()
print(f"\nHKA después de corrección: positivos={pos} negativos={neg}")
