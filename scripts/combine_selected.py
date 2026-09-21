"""Combina round_selected_data.csv + round_selected.csv y copia las imagenes."""
import pandas as pd
import shutil
from pathlib import Path

reports = Path("reports/validation_round")

# Leer
a = pd.read_csv(reports / "round_selected_data.csv")
b = pd.read_csv(reports / "round_selected.csv")
combined = pd.concat([a, b], ignore_index=True).drop_duplicates(subset=["case_id"], keep="last")
print(f"Combinado: {len(combined)} registros (data={len(a)} + TeleRx={len(b)})")

# Guardar CSV + XLSX
combined.to_csv(reports / "round_selected_combined.csv", index=False, encoding="utf-8")
combined.to_excel(reports / "round_selected_combined.xlsx", index=False)
print("CSV y XLSX guardados")

# Copiar imagenes
out_dir = reports / "selected_images"
out_dir.mkdir(parents=True, exist_ok=True)

copied = 0
missing = 0
for _, row in combined.iterrows():
    src = Path(str(row["image_fullpath"]))
    if src.exists():
        dest = out_dir / src.name
        shutil.copy2(src, dest)
        copied += 1
    else:
        print(f"  FALTA: {src}")
        missing += 1

print(f"Imagenes copiadas: {copied}")
if missing:
    print(f"Faltantes: {missing}")
