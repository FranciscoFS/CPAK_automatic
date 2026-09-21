"""Extrae métricas de los JSON generados por batch inference al mediciones.xlsx.
Corregido: busca por ID estudio primero, RUT como fallback.
"""
import json
from pathlib import Path

import pandas as pd

med = pd.read_excel("reports/validation_round/mediciones.xlsx")
jsons_dir = Path("visualizations/final_pipeline_batch")

# Cargar todos los JSONs una sola vez con su image path
json_data = []
for jf in jsons_dir.glob("*.json"):
    with open(jf, encoding="utf-8") as f:
        data = json.load(f)
    json_data.append((jf.stem, data))

print(f"JSONs disponibles: {len(json_data)}")

ok = 0
fail = 0
for i, row in med.iterrows():
    rut = str(row["RUT"]).strip()
    id_est = str(row["ID estudio"]).strip()
    found = False

    # Estrategia: buscar match por ID estudio (más específico)
    for jname, data in json_data:
        img_path = data.get("image", "")
        match = False
        # Prioridad 1: ID estudio en el path (ej: rx_RUT_IDestudio_...)
        if id_est in img_path:
            match = True
        # Prioridad 2: solo RUT si no hay ID estudio en ningún JSON
        # (pctes con un solo estudio no tienen ID en el nombre)
        if not match and rut in img_path:
            # Solo aceptar si este RUT NO tiene JSONs con ID estudio
            has_id_json = any(id_est in jn2 for jn2, _ in json_data if rut in jn2)
            if not has_id_json:
                match = True

        if match:
            sides = data.get("sides", {})
            for lado_key, col_hka, col_ldfa, col_mpta in [
                ("Der", "HKA_Der", "mLDFA_Der", "mMPTA_Der"),
                ("Izq", "HKA_Izq", "mLDFA_Izq", "mMPTA_Izq"),
            ]:
                lado = sides.get(lado_key, {})
                metrics = lado.get("metrics", {})
                if metrics.get("status") == "ok":
                    if metrics.get("HKA") is not None:
                        med.at[i, col_hka] = round(metrics["HKA"], 2)
                    if metrics.get("LDFA") is not None:
                        med.at[i, col_ldfa] = round(metrics["LDFA"], 2)
                    if metrics.get("MPTA") is not None:
                        med.at[i, col_mpta] = round(metrics["MPTA"], 2)
            found = True
            ok += 1
            break
    if not found:
        fail += 1
        print(f"  NO MATCH: RUT={rut}  ID_estudio={id_est}")

med.to_excel("reports/validation_round/mediciones.xlsx", index=False)

# También guardar copia como mediciones_IA.xlsx (solo columnas relevantes)
ia_cols = ['RUT', 'ID estudio', 'HKA_Der', 'mLDFA_Der', 'mMPTA_Der',
           'HKA_Izq', 'mLDFA_Izq', 'mMPTA_Izq']
med[ia_cols].to_excel(
    "reports/validation_round/Mediciones/mediciones_IA.xlsx",
    index=False
)

print(f"Mediciones extraidas: {ok} imagenes | No encontradas: {fail}")
print(f"Registros con HKA Der: {med['HKA_Der'].notna().sum()}")
print(f"Registros con HKA Izq: {med['HKA_Izq'].notna().sum()}")
print(f"Registros con mLDFA Der: {med['mLDFA_Der'].notna().sum()}")
print(f"Registros con mLDFA Izq: {med['mLDFA_Izq'].notna().sum()}")
print(f"Registros con mMPTA Der: {med['mMPTA_Der'].notna().sum()}")
print(f"Registros con mMPTA Izq: {med['mMPTA_Izq'].notna().sum()}")
