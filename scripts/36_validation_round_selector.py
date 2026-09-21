"""
Selector/etiquetador de ronda de validacion para TeleRx.

Objetivo:
- Navegar imagenes de Test_Data_Historic/TeleRx_validas.
- Marcar inclusion en ronda (>=50 casos).
- Etiquetar por lado: protesis y grado KL.
- Guardar tabla trazable (imagen, RUT/ID, fechas del log cuando existan).

Uso:
  conda activate physis_seg
  streamlit run scripts/36_validation_round_selector.py
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import streamlit as st
from PIL import Image


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_IMAGES_DIR = BASE_DIR / "data"
DEFAULT_LOG_CSV = BASE_DIR / "data" / "registro_inferi_con_edad.xlsx"
DATA_IMAGES_DIR = BASE_DIR / "data"
DATA_METADATA_PATH = BASE_DIR / "data" / "registro_inferi_con_edad.xlsx"
TELERX_IMAGES_DIR = BASE_DIR / "Test_Data_Historic" / "TeleRx_validas"
TELERX_METADATA_PATH = BASE_DIR / "Test_Data_Historic" / "descargas_log.csv"
OUTPUT_DIR = BASE_DIR / "reports" / "validation_round"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
KL_OPTIONS = ["NA", "0", "1", "2", "3", "4"]

WIDGET_KEYS = {
    "selected_round": "selected_round_input",
    "prosthesis_der": "prosthesis_der_input",
    "prosthesis_izq": "prosthesis_izq_input",
    "kl_der": "kl_der_input",
    "kl_izq": "kl_izq_input",
    "reviewer": "reviewer_input",
    "notes": "notes_input",
}

SOURCE_CONFIGS = {
    "data": {
        "images_dir": DATA_IMAGES_DIR,
        "metadata_path": DATA_METADATA_PATH,
    },
    "TeleRx_validas": {
        "images_dir": TELERX_IMAGES_DIR,
        "metadata_path": TELERX_METADATA_PATH,
    },
}


def as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    s = str(value).strip().lower()
    return s in {"true", "1", "yes", "y", "si", "sí"}


def sanitize_source_name(images_dir: Path) -> str:
    name = images_dir.name.strip() or "catalog"
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", name)
    return safe or "catalog"


def get_output_paths(images_dir: Path) -> tuple[Path, Path, Path]:
    source_tag = sanitize_source_name(images_dir)
    annotations_csv = OUTPUT_DIR / f"round_annotations_{source_tag}.csv"
    selected_csv = OUTPUT_DIR / f"round_selected_{source_tag}.csv"
    selected_xlsx = OUTPUT_DIR / f"round_selected_{source_tag}.xlsx"
    return annotations_csv, selected_csv, selected_xlsx


def parse_image_ids(image_name: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extrae IDs candidatos desde nombres tipo:
      rx_10015390-4_8813650.png
      rx_10055981-1_8434782 (1).png
      rx_10089376_10089376.png
    Retorna: (rut_candidate, study_candidate)
    """
    stem = Path(image_name).stem
    stem = re.sub(r"\s*\(\d+\)$", "", stem)  # quita sufijo (1), (2), etc.

    m = re.match(r"^rx_([^_]+)_([^_]+)$", stem, flags=re.IGNORECASE)
    if m:
        rut_c = m.group(1).strip()
        study_c = m.group(2).strip()
        return rut_c or None, study_c or None

    # fallback: primer token despues de rx_
    m2 = re.match(r"^rx_([^_]+)$", stem, flags=re.IGNORECASE)
    if m2:
        tok = m2.group(1).strip()
        return tok or None, tok or None

    return None, None


@st.cache_data(show_spinner=False)
def list_images(images_dir: Path) -> list[Path]:
    if not images_dir.exists():
        return []
    return sorted([p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in VALID_EXTS])


@st.cache_data(show_spinner=False)
def load_metadata_table(metadata_path: Path) -> pd.DataFrame:
    if not metadata_path.exists():
        return pd.DataFrame()

    if metadata_path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(metadata_path)
    else:
        df = pd.read_csv(metadata_path)

    for col in ["timestamp_start", "timestamp_end", "rut", "filename", "status", "fecha_estudio", "edad", "sexo", "nombre", "fecha_nac"]:
        if col not in df.columns:
            df[col] = None

    if "filename" in df.columns:
        df["study_id"] = (
            df["filename"].astype(str).str.replace(".zip", "", regex=False).str.strip()
        )
    else:
        df["study_id"] = None

    df["rut"] = df["rut"].astype(str).str.strip().str.upper()
    return df


def resolve_log_match(log_df: pd.DataFrame, rut_c: Optional[str], study_c: Optional[str]) -> dict:
    """
    Resuelve metadata desde log con prioridad:
    1) study_id exacto
    2) rut exacto (si existe)
    Toma la primera fila por timestamp_start ascendente para estabilidad.
    """
    if log_df.empty:
        return {
            "log_match_type": "none",
            "log_rut": None,
            "log_study_id": None,
            "log_timestamp_start": None,
            "log_timestamp_end": None,
            "log_status": None,
            "log_nombre": None,
            "log_sexo": None,
            "log_edad": None,
            "log_fecha_nac": None,
            "log_fecha_estudio": None,
        }

    match = pd.DataFrame()
    match_type = "none"

    if study_c:
        cand = log_df[log_df["study_id"] == str(study_c)]
        if not cand.empty:
            match = cand.sort_values("timestamp_start", ascending=True)
            match_type = "study_id"

    if match.empty and rut_c:
        cand = log_df[log_df["rut"] == str(rut_c)]
        if not cand.empty:
            match = cand.sort_values("timestamp_start", ascending=True)
            match_type = "rut"

    if match.empty:
        return {
            "log_match_type": "none",
            "log_rut": None,
            "log_study_id": None,
            "log_timestamp_start": None,
            "log_timestamp_end": None,
            "log_status": None,
            "log_nombre": None,
            "log_sexo": None,
            "log_edad": None,
            "log_fecha_nac": None,
            "log_fecha_estudio": None,
        }

    row = match.iloc[0]
    return {
        "log_match_type": match_type,
        "log_rut": row.get("rut"),
        "log_study_id": row.get("study_id"),
        "log_timestamp_start": row.get("timestamp_start"),
        "log_timestamp_end": row.get("timestamp_end"),
        "log_status": row.get("status"),
        "log_nombre": row.get("nombre"),
        "log_sexo": row.get("sexo"),
        "log_edad": row.get("edad"),
        "log_fecha_nac": row.get("fecha_nac"),
        "log_fecha_estudio": row.get("fecha_estudio"),
    }


def default_record_for_image(img_path: Path, images_dir: Path, log_df: pd.DataFrame) -> dict:
    rut_c, study_c = parse_image_ids(img_path.name)
    meta = resolve_log_match(log_df, rut_c, study_c)

    return {
        "case_id": img_path.stem,
        "image_name": img_path.name,
        "image_relpath": str(img_path.relative_to(images_dir)),
        "image_fullpath": str(img_path),
        "selected_round": False,
        "prosthesis_der": False,
        "prosthesis_izq": False,
        "kl_der": "NA",
        "kl_izq": "NA",
        "measurement_hka_der": None,
        "measurement_hka_izq": None,
        "measurement_mldfa_der": None,
        "measurement_mldfa_izq": None,
        "measurement_mptra_der": None,
        "measurement_mptra_izq": None,
        "reviewer": "",
        "notes": "",
        "id_rut_candidate": rut_c,
        "id_study_candidate": study_c,
        "file_mtime": img_path.stat().st_mtime,
        **meta,
    }


def save_annotations(df: pd.DataFrame, annotations_csv: Path) -> None:
    """Guarda el catalogo completo (trabajo) en round_annotations.csv."""
    df.to_csv(annotations_csv, index=False, encoding="utf-8")


def export_selected(df: pd.DataFrame, selected_csv: Path, selected_xlsx: Path) -> tuple[Optional[Path], Optional[Path]]:
    """Exporta solo los seleccionados a round_selected.csv y .xlsx."""
    selected = df[df["selected_round"].apply(lambda v: str(v).lower() in ("true", "1", "yes"))].copy()
    csv_path: Optional[Path] = None
    xlsx_path: Optional[Path] = None
    if not selected.empty:
        selected.to_csv(selected_csv, index=False, encoding="utf-8")
        csv_path = selected_csv
        try:
            selected.to_excel(selected_xlsx, index=False)
            xlsx_path = selected_xlsx
        except Exception:
            pass
    else:
        # Si no hay seleccionados, dejar el archivo en estado vacio consistente.
        empty = df.iloc[0:0].copy()
        empty.to_csv(selected_csv, index=False, encoding="utf-8")
        csv_path = selected_csv
    return csv_path, xlsx_path


def read_csv_safe(csv_path: Path) -> pd.DataFrame:
    """Lee CSV y devuelve DataFrame vacio si el archivo existe pero no tiene columnas."""
    if not csv_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def sanitize_catalog_df(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina columnas duplicadas y residuos de merges previos (__sel, __sel.1, etc.)."""
    if df.empty:
        return df

    clean_df = df.loc[:, ~df.columns.duplicated()].copy()
    keep_mask = ~clean_df.columns.to_series().str.contains(r"__sel(?:\.\d+)?$", regex=True)
    clean_df = clean_df.loc[:, keep_mask]
    return clean_df


@st.cache_data(show_spinner=False)
def load_image_for_display(image_path: str, max_width: int = 1400) -> Image.Image:
    """Carga y reduce imagen para visualizacion rapida en UI."""
    img = Image.open(image_path)
    if img.width > max_width:
        ratio = max_width / float(img.width)
        new_size = (max_width, int(img.height * ratio))
        img = img.resize(new_size, Image.Resampling.BILINEAR)
    return img


def load_or_init_catalog(
    images: list[Path],
    images_dir: Path,
    log_df: pd.DataFrame,
    annotations_csv: Path,
    selected_csv: Path,
) -> pd.DataFrame:
    selected_df = sanitize_catalog_df(read_csv_safe(selected_csv))

    def apply_selected_snapshot(base_df: pd.DataFrame) -> pd.DataFrame:
        base_df = sanitize_catalog_df(base_df)
        if selected_df.empty or "image_fullpath" not in base_df.columns:
            return base_df

        selected_columns = [
            "image_fullpath",
            "selected_round",
            "prosthesis_der",
            "prosthesis_izq",
            "kl_der",
            "kl_izq",
            "reviewer",
            "notes",
        ]
        available_selected_columns = [col for col in selected_columns if col in selected_df.columns]
        selected_by_path = selected_df[available_selected_columns].copy()
        if "selected_round" in selected_by_path.columns:
            selected_by_path = selected_by_path[selected_by_path["selected_round"].apply(as_bool)].copy()

        if selected_by_path.empty or "image_fullpath" not in selected_by_path.columns:
            return base_df

        selected_by_path["image_fullpath"] = selected_by_path["image_fullpath"].astype(str)
        selected_by_path = selected_by_path.drop_duplicates(subset=["image_fullpath"], keep="last")

        merged = base_df.merge(
            selected_by_path,
            on="image_fullpath",
            how="left",
            suffixes=("", "__sel"),
        )

        cols_to_sync = [
            "selected_round",
            "prosthesis_der",
            "prosthesis_izq",
            "kl_der",
            "kl_izq",
            "reviewer",
            "notes",
        ]
        for col in cols_to_sync:
            sel_col = f"{col}__sel"
            if sel_col in merged.columns:
                merged[col] = merged[sel_col].combine_first(merged[col])
                merged.drop(columns=[sel_col], inplace=True)

        for col in ["selected_round", "prosthesis_der", "prosthesis_izq"]:
            if col in merged.columns:
                merged[col] = merged[col].apply(as_bool)

        return merged

    if annotations_csv.exists():
        df = sanitize_catalog_df(pd.read_csv(annotations_csv))
        st.caption(f"Cargado desde: {annotations_csv.name}")
        if "selected_round" in df.columns:
            df["selected_round"] = df["selected_round"].apply(as_bool)
        for col in ["prosthesis_der", "prosthesis_izq"]:
            if col in df.columns:
                df[col] = df[col].apply(as_bool)
        # Agrega imagenes nuevas que no esten en el catalogo
        known = set(df["image_fullpath"].astype(str).tolist())
        new_rows = []
        for p in images:
            if str(p) not in known:
                new_rows.append(default_record_for_image(p, images_dir, log_df))
        if new_rows:
            df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        df = apply_selected_snapshot(df)
        save_annotations(df, annotations_csv)
        return df

    rows = [default_record_for_image(p, images_dir, log_df) for p in images]
    df = sanitize_catalog_df(pd.DataFrame(rows))
    df = apply_selected_snapshot(df)
    save_annotations(df, annotations_csv)
    st.caption(f"Catalogo creado: {annotations_csv.name} ({len(df)} casos)")
    return df


def build_case_order_by_rut(df: pd.DataFrame, seed: Optional[int] = None) -> list[int]:
    """
    Construye un orden de casos agrupado por RUT y barajado por RUT,
    para que no se acumulen imagenes del mismo paciente seguidas.
    """
    rng = random.Random(seed) if seed is not None else random.Random()

    # Agrupar indices por RUT (los que no tienen RUT van en grupo separado)
    rut_groups: dict[str, list[int]] = {}
    no_rut_indices: list[int] = []
    for i in range(len(df)):
        rut = str(df.iloc[i].get("id_rut_candidate", ""))
        if rut and rut.lower() != "none" and rut != "nan":
            rut_groups.setdefault(rut, []).append(i)
        else:
            no_rut_indices.append(i)

    # Barajar lista de RUTs
    rut_keys = list(rut_groups.keys())
    rng.shuffle(rut_keys)

    # Aplanar: primero todos los grupos de RUT, luego los que no tienen RUT
    order: list[int] = []
    for key in rut_keys:
        order.extend(rut_groups[key])
    rng.shuffle(no_rut_indices)
    order.extend(no_rut_indices)

    return order


def ensure_state(df: pd.DataFrame) -> None:
    if "active_catalog_key" not in st.session_state:
        st.session_state.active_catalog_key = None
    catalog_key = st.session_state.get("pending_catalog_key")

    if st.session_state.active_catalog_key != catalog_key:
        st.session_state.annot_df = df
        st.session_state.idx = 0
        st.session_state.case_order = build_case_order_by_rut(df, seed=20260712)
        st.session_state.loaded_case_idx = None
        st.session_state.active_catalog_key = catalog_key
        return

    if "annot_df" not in st.session_state:
        st.session_state.annot_df = df
    if "idx" not in st.session_state:
        st.session_state.idx = 0
    if "case_order" not in st.session_state or len(st.session_state.case_order) != len(df):
        st.session_state.case_order = build_case_order_by_rut(df, seed=20260712)
    if "loaded_case_idx" not in st.session_state:
        st.session_state.loaded_case_idx = None


def clamp_index() -> None:
    n = len(st.session_state.annot_df)
    if n == 0:
        st.session_state.idx = 0
        return
    st.session_state.idx = max(0, min(st.session_state.idx, n - 1))


def sync_widgets_from_row(row: pd.Series) -> None:
    st.session_state[WIDGET_KEYS["selected_round"]] = as_bool(row["selected_round"])
    st.session_state[WIDGET_KEYS["prosthesis_der"]] = as_bool(row["prosthesis_der"])
    st.session_state[WIDGET_KEYS["prosthesis_izq"]] = as_bool(row["prosthesis_izq"])
    st.session_state[WIDGET_KEYS["kl_der"]] = str(row["kl_der"]) if str(row["kl_der"]) in KL_OPTIONS else "NA"
    st.session_state[WIDGET_KEYS["kl_izq"]] = str(row["kl_izq"]) if str(row["kl_izq"]) in KL_OPTIONS else "NA"
    st.session_state[WIDGET_KEYS["reviewer"]] = "" if pd.isna(row.get("reviewer")) else str(row.get("reviewer", ""))
    st.session_state[WIDGET_KEYS["notes"]] = "" if pd.isna(row.get("notes")) else str(row.get("notes", ""))


def save_current_case(actual_idx: int, annotations_csv: Path) -> None:
    st.session_state.annot_df.at[actual_idx, "selected_round"] = bool(st.session_state[WIDGET_KEYS["selected_round"]])
    st.session_state.annot_df.at[actual_idx, "prosthesis_der"] = bool(st.session_state[WIDGET_KEYS["prosthesis_der"]])
    st.session_state.annot_df.at[actual_idx, "prosthesis_izq"] = bool(st.session_state[WIDGET_KEYS["prosthesis_izq"]])
    st.session_state.annot_df.at[actual_idx, "kl_der"] = st.session_state[WIDGET_KEYS["kl_der"]]
    st.session_state.annot_df.at[actual_idx, "kl_izq"] = st.session_state[WIDGET_KEYS["kl_izq"]]
    st.session_state.annot_df.at[actual_idx, "reviewer"] = st.session_state[WIDGET_KEYS["reviewer"]].strip()
    st.session_state.annot_df.at[actual_idx, "notes"] = st.session_state[WIDGET_KEYS["notes"]].strip()
    save_annotations(st.session_state.annot_df, annotations_csv)
    st.session_state.last_action_message = "Caso guardado"


def move_case(delta: int, annotations_csv: Path) -> None:
    current_actual_idx = st.session_state.case_order[st.session_state.idx]
    save_current_case(current_actual_idx, annotations_csv)
    st.session_state.idx += delta
    clamp_index()
    st.session_state.loaded_case_idx = None
    if delta > 0:
        st.session_state.last_action_message = "Caso guardado y avanzado"
    elif delta < 0:
        st.session_state.last_action_message = "Caso guardado y retrocedido"


def jump_to_next_unlabeled(annotations_csv: Path) -> None:
    """Salta al siguiente caso no marcado como selected_round."""
    current_pos = st.session_state.idx
    order = st.session_state.case_order
    n = len(order)
    current_actual_idx = order[current_pos]
    save_current_case(current_actual_idx, annotations_csv)

    for step in range(1, n + 1):
        pos = (current_pos + step) % n
        actual_idx = order[pos]
        row = st.session_state.annot_df.iloc[actual_idx]
        if not as_bool(row.get("selected_round")):
            st.session_state.idx = pos
            st.session_state.loaded_case_idx = None
            st.session_state.last_action_message = "Salto al siguiente no etiquetado"
            return

    st.session_state.last_action_message = "No hay casos pendientes (todos etiquetados)"


def reshuffle_cases() -> None:
    current_actual_idx = st.session_state.case_order[st.session_state.idx]
    new_order = build_case_order_by_rut(st.session_state.annot_df)
    st.session_state.case_order = new_order
    try:
        st.session_state.idx = new_order.index(current_actual_idx)
    except ValueError:
        st.session_state.idx = 0
    st.session_state.loaded_case_idx = None
    st.session_state.last_action_message = "Orden aleatorio actualizado"


def main() -> None:
    st.set_page_config(page_title="CPAK · Selector Ronda Validacion", layout="wide")
    st.title("Selector de casos · Validacion TeleRx")

    with st.sidebar:
        st.markdown("### Fuente")
        source_name = st.selectbox("Dataset", options=list(SOURCE_CONFIGS.keys()), index=0)
        source_cfg = SOURCE_CONFIGS[source_name]
        images_dir_str = st.text_input("Carpeta de imagenes", value=str(source_cfg["images_dir"]))
        log_csv_str = st.text_input("Metadata (CSV/XLSX)", value=str(source_cfg["metadata_path"]))

        st.markdown("### Meta")
        target_n = st.number_input("Meta de casos", min_value=1, max_value=500, value=50, step=1)

        st.markdown("### Orden")
        st.caption("Orden aleatorio por RUT (agrupa imagenes del mismo paciente).")
        reshuffle_clicked = st.button("Reordenar aleatoriamente", use_container_width=True)

    images_dir = Path(images_dir_str)
    log_csv = Path(log_csv_str)
    annotations_csv, selected_csv, selected_xlsx = get_output_paths(images_dir)
    st.session_state.pending_catalog_key = str(annotations_csv)

    images = list_images(images_dir)
    log_df = load_metadata_table(log_csv)

    if not images:
        st.error(f"No se encontraron imagenes en: {images_dir}")
        return

    df = load_or_init_catalog(images, images_dir, log_df, annotations_csv, selected_csv)
    ensure_state(df)
    clamp_index()

    if reshuffle_clicked:
        reshuffle_cases()

    # Resumen
    dfa = st.session_state.annot_df
    selected_n = int(dfa["selected_round"].apply(as_bool).sum())
    actual_idx = st.session_state.case_order[st.session_state.idx]
    row = dfa.iloc[actual_idx].copy()

    if st.session_state.loaded_case_idx != actual_idx:
        sync_widgets_from_row(row)
        st.session_state.loaded_case_idx = actual_idx

    last_action_message = st.session_state.pop("last_action_message", None)
    if last_action_message:
        st.success(last_action_message)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total imagenes", len(dfa))
    c2.metric("Seleccionadas", selected_n)
    c3.metric("Faltan para meta", max(0, int(target_n) - selected_n))

    # Navegacion
    nav1, nav2, nav3, nav4, nav5, nav6 = st.columns([1, 1, 1, 1, 2, 2])
    if nav1.button("← Anterior", use_container_width=True):
        move_case(-1, annotations_csv)
    if nav2.button("Siguiente →", use_container_width=True):
        move_case(1, annotations_csv)
    if nav3.button("+10", use_container_width=True):
        move_case(10, annotations_csv)
    if nav4.button("+50", use_container_width=True):
        move_case(50, annotations_csv)
    if nav5.button("Siguiente pendiente", use_container_width=True):
        jump_to_next_unlabeled(annotations_csv)

    jump = nav6.number_input(
        "Ir a indice",
        min_value=1,
        max_value=len(dfa),
        value=int(st.session_state.idx) + 1,
        step=1,
    )
    if nav6.button("Ir", use_container_width=True):
        save_current_case(actual_idx, annotations_csv)
        st.session_state.idx = int(jump) - 1
        clamp_index()
        st.session_state.loaded_case_idx = None
        st.session_state.last_action_message = "Caso guardado y salto aplicado"

    left, right = st.columns([2.2, 1.3])
    with left:
        st.markdown(f"### Caso {st.session_state.idx + 1} / {len(dfa)}")
        img = load_image_for_display(str(row["image_fullpath"]))
        st.image(img, use_container_width=True)

    with right:
        st.markdown("### Etiquetado")
        if as_bool(row["selected_round"]):
            st.success("Este caso ya esta agregado a la ronda.")
        else:
            st.warning("Este caso aun no esta agregado a la ronda.")
        with st.form("case_annotation_form", clear_on_submit=False):
            st.checkbox("Incluir en ronda", key=WIDGET_KEYS["selected_round"])

            st.markdown("**Protesis**")
            st.checkbox("Der", key=WIDGET_KEYS["prosthesis_der"])
            st.checkbox("Izq", key=WIDGET_KEYS["prosthesis_izq"])

            st.markdown("**KL**")
            st.selectbox("KL Der", options=KL_OPTIONS, key=WIDGET_KEYS["kl_der"])
            st.selectbox("KL Izq", options=KL_OPTIONS, key=WIDGET_KEYS["kl_izq"])

            st.text_input("Reviewer", key=WIDGET_KEYS["reviewer"])
            st.text_area("Notas", key=WIDGET_KEYS["notes"], height=100)

            st.markdown("### Metadata")
            st.text(f"image_name: {row['image_name']}")
            st.text(f"id_rut_candidate: {row.get('id_rut_candidate')}")
            st.text(f"id_study_candidate: {row.get('id_study_candidate')}")
            st.text(f"log_match_type: {row.get('log_match_type')}")
            st.text(f"log_rut: {row.get('log_rut')}")
            st.text(f"log_study_id: {row.get('log_study_id')}")
            st.text(f"log_timestamp_start: {row.get('log_timestamp_start')}")
            st.text(f"log_nombre: {row.get('log_nombre')}")
            st.text(f"log_sexo: {row.get('log_sexo')}")
            st.text(f"log_edad: {row.get('log_edad')}")
            st.text(f"log_fecha_estudio: {row.get('log_fecha_estudio')}")
            st.caption(f"Catalogo: {annotations_csv.name} | Export: {selected_csv.name}")

            action1, action2 = st.columns(2)
            action1.form_submit_button(
                "Guardar etiqueta del caso",
                type="primary",
                use_container_width=True,
                on_click=save_current_case,
                args=(actual_idx, annotations_csv),
            )
            action2.form_submit_button(
                "Guardar y siguiente",
                use_container_width=True,
                on_click=move_case,
                args=(1, annotations_csv),
            )

    st.markdown("---")
    st.markdown("### Exportar seleccionadas")

    e1, e2 = st.columns([1, 2])
    if e1.button("Exportar a CSV + Excel", use_container_width=True):
        save_annotations(st.session_state.annot_df, annotations_csv)
        csv_out, xlsx_out = export_selected(st.session_state.annot_df, selected_csv, selected_xlsx)
        if csv_out is not None:
            if xlsx_out is not None:
                e2.success(f"Exportado: {csv_out.name} y {xlsx_out.name}")
            else:
                e2.success(f"Exportado: {csv_out.name}")
        else:
            e2.warning("No hay casos seleccionados para exportar.")

    # Preview: mezcla exportado previo + seleccionados de la sesion actual
    with st.expander("Preview seleccionadas", expanded=False):
        from_file = pd.DataFrame()
        if selected_csv.exists():
            from_file = sanitize_catalog_df(read_csv_safe(selected_csv))
            # Defensa: aunque el CSV historico tenga todos los casos, el preview solo muestra seleccionados.
            if "selected_round" in from_file.columns:
                from_file = from_file[from_file["selected_round"].apply(as_bool)].copy()

        in_session = sanitize_catalog_df(
            st.session_state.annot_df[st.session_state.annot_df["selected_round"].apply(as_bool)].copy()
        )

        if from_file.empty and in_session.empty:
            st.caption("Sin seleccionados por ahora.")
        else:
            if "case_id" in from_file.columns and "case_id" in in_session.columns:
                preview_df = pd.concat([from_file, in_session], ignore_index=True)
                preview_df = preview_df.drop_duplicates(subset=["case_id"], keep="last")
            else:
                preview_df = pd.concat([from_file, in_session], ignore_index=True).drop_duplicates(keep="last")

            preview_df = preview_df.reset_index(drop=True)

            st.caption(
                f"Exportados previos: {len(from_file)} | Seleccionados en sesion: {len(in_session)} | Preview total: {len(preview_df)}"
            )
            st.dataframe(
                preview_df[
                    [
                        "case_id",
                        "image_name",
                        "id_rut_candidate",
                        "id_study_candidate",
                        "log_timestamp_start",
                        "selected_round",
                        "prosthesis_der",
                        "prosthesis_izq",
                        "kl_der",
                        "kl_izq",
                        "measurement_hka_der",
                        "measurement_hka_izq",
                        "measurement_mldfa_der",
                        "measurement_mldfa_izq",
                        "measurement_mptra_der",
                        "measurement_mptra_izq",
                        "reviewer",
                    ]
                ],
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
