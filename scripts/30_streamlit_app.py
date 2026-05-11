"""
CPAK - Streamlit Web UI
Análisis clínico de alineación femoral (LDFA, MPTA, aHKA, JLO).

Launch:
    conda run -n physis_seg streamlit run scripts/30_streamlit_app.py
"""
import importlib.util
import json
import tempfile
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
PIPELINE_PATH = BASE_DIR / "scripts" / "13_final_inference.py"

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CPAK · Inferencia Clínica",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS tema profesional
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
/* Fondo principal más oscuro */
[data-testid="stAppViewContainer"] {
    background-color: #0f1117;
}
[data-testid="stSidebar"] {
    background-color: #161b27;
}

/* Métricas más grandes */
[data-testid="stMetricValue"] {
    font-size: 24px !important;
    font-weight: 700 !important;
}
[data-testid="stMetricLabel"] {
    font-size: 12px !important;
    color: #8b9ab1 !important;
}

/* Título principal */
.cpak-title {
    font-size: 32px;
    font-weight: 800;
    background: linear-gradient(90deg, #4f9cf9, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 4px;
}
.cpak-subtitle {
    font-size: 14px;
    color: #8b9ab1;
    margin-bottom: 24px;
}

/* Tarjeta de clasificación CPAK */
.cpak-badge {
    padding: 10px 18px;
    border-radius: 8px;
    text-align: center;
    font-size: 16px;
    font-weight: 700;
    letter-spacing: 0.5px;
    margin-top: 8px;
    margin-bottom: 16px;
}

/* Separador lateral */
.sidebar-section {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #4f9cf9;
    margin-top: 20px;
    margin-bottom: 6px;
}

/* Panel de detecciones */
.det-pill {
    display: inline-block;
    background: #1e2535;
    color: #c8d0e0;
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 12px;
    margin: 2px;
}
.det-missing {
    color: #e06c75;
}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Load pipeline module (cached for lifetime of server)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Cargando pipeline…")
def load_pipeline():
    if not PIPELINE_PATH.exists():
        raise FileNotFoundError(f"Pipeline no encontrado: {PIPELINE_PATH}")
    spec = importlib.util.spec_from_file_location("cpak_pipeline", str(PIPELINE_PATH))
    if spec is None or spec.loader is None:
        raise RuntimeError("No se pudo cargar el módulo del pipeline")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@st.cache_resource(show_spinner="Cargando modelo de detección…")
def load_det_model(path: str):
    return pipeline.YOLO(path)


@st.cache_resource(show_spinner="Cargando modelo de pose…")
def load_pose_model(path: str):
    return pipeline.YOLO(path)


pipeline = load_pipeline()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def cv2_to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def fmt(v, d=2):
    return "N/A" if v is None else f"{v:.{d}f}"


def cpak_color(cpak: str) -> str:
    if "Varo" in cpak:
        return "#e06c75"
    if "Valgo" in cpak:
        return "#61afef"
    return "#98c379"


def render(image_path: Path, payload: dict, boxes: bool, axes: bool) -> Image.Image:
    canvas = pipeline.render_overlay_from_payload(
        image_path=image_path,
        payload=payload,
        draw_boxes=boxes,
        draw_axes=axes,
    )
    return cv2_to_pil(canvas)


def get_joint_crop(image_path: Path, payload: dict, side: str, joint: str, margin_px: int = 30) -> Image.Image | None:
    """Recorta la zona de un joint específico con margen adicional."""
    for det in payload.get("detections", {}).values():
        if det["joint"] == joint and det["side"] == side:
            img = cv2.imread(str(image_path))
            h, w = img.shape[:2]
            x1, y1, x2, y2 = [int(v) for v in det["xyxy"]]
            x1 = max(0, x1 - margin_px)
            y1 = max(0, y1 - margin_px)
            x2 = min(w, x2 + margin_px)
            y2 = min(h, y2 + margin_px)
            crop = img[y1:y2, x1:x2]
            return cv2_to_pil(crop)
    return None


# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
if "payload" not in st.session_state:
    st.session_state.payload = None
    st.session_state.image_path = None
    st.session_state.ran_once = False

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-section">Imagen</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Seleccionar Rx",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"],
        label_visibility="collapsed",
    )

    st.markdown('<div class="sidebar-section">Modelos</div>', unsafe_allow_html=True)
    det_path = st.text_input("Detección (.pt)", value=str(pipeline.DEFAULT_DET_MODEL))
    pose_path = st.text_input("Pose (.pt)", value=str(pipeline.DEFAULT_POSE_MODEL))

    st.markdown('<div class="sidebar-section">Parámetros</div>', unsafe_allow_html=True)
    det_conf = st.slider("Detection conf", 0.0, 1.0, 0.20, 0.05)
    pose_conf = st.slider("Pose conf", 0.0, 1.0, 0.20, 0.05)
    padding = st.slider("Padding BB (%)", 0.0, 0.50, 0.15, 0.05)

    st.markdown('<div class="sidebar-section">Capas de visualización</div>', unsafe_allow_html=True)
    show_boxes = st.checkbox("Bounding Boxes", value=True)
    show_axes = st.checkbox("Ejes anatómicos", value=True)

    st.markdown("---")
    run_btn = st.button("▶  Ejecutar inferencia", use_container_width=True, type="primary")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<div class="cpak-title">🦴 CPAK · Inferencia Clínica</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="cpak-subtitle">Detección de zonas · Keypoints anatómicos · LDFA / MPTA / aHKA / JLO</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Run inference
# ---------------------------------------------------------------------------
if run_btn:
    if uploaded is None:
        st.error("Carga una imagen Rx primero.")
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
            tmp.write(uploaded.read())
            tmp_path = Path(tmp.name)

        try:
            with st.spinner("Ejecutando pipeline…"):
                det_model = load_det_model(det_path)
                pose_model = load_pose_model(pose_path)
                out_dir = Path(tempfile.gettempdir()) / "cpak_streamlit"
                _, _, payload = pipeline.process_image(
                    det_model=det_model,
                    pose_model=pose_model,
                    image_path=tmp_path,
                    output_dir=out_dir,
                    det_conf=det_conf,
                    pose_conf=pose_conf,
                    padding=padding,
                    draw_boxes=show_boxes,
                    draw_axes=show_axes,
                )
            st.session_state.payload = payload
            st.session_state.image_path = tmp_path
            st.session_state.ran_once = True
            st.success("Inferencia completada ✓")
        except Exception as exc:
            st.error(f"Error: {exc}")

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
if st.session_state.ran_once and st.session_state.payload is not None:
    payload = st.session_state.payload
    img_path = st.session_state.image_path

    # Re-render on every interaction (instant, no re-inference)
    overlay = render(img_path, payload, show_boxes, show_axes)

    # ── Layout: métricas | imagen | controles ──
    col_left, col_center, col_right = st.columns([1, 2.2, 1], gap="large")

    # ── Métricas ──────────────────────────────────────────────────────────
    with col_left:
        st.markdown("### 📊 Métricas")

        for side in ["Der", "Izq"]:
            side_color = "#98c379" if side == "Der" else "#61afef"
            st.markdown(
                f'<span style="font-size:16px;font-weight:700;color:{side_color};">{side}</span>',
                unsafe_allow_html=True,
            )
            s = payload.get("sides", {}).get(side, {})
            m = s.get("metrics", {})

            if m.get("status") == "ok":
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("LDFA °", fmt(m.get("LDFA")))
                    st.metric("aHKA °", fmt(m.get("aHKA")))
                with c2:
                    st.metric("MPTA °", fmt(m.get("MPTA")))
                    st.metric("JLO °", fmt(m.get("JLO")))

                cpak = m.get("CPAK", "")
                cpak_type = m.get("CPAK_type", "?")
                color = cpak_color(cpak)
                st.markdown(
                    f'<div class="cpak-badge" style="background:{color}22;'
                    f'border:1.5px solid {color};color:{color};">'
                    f'Tipo {cpak_type} &nbsp;·&nbsp; {cpak}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.warning(f"⚠ {m.get('status', 'datos incompletos')}")

    # ── Imagen central ─────────────────────────────────────────────────────
    with col_center:
        st.markdown("### 🖼️ Overlay")
        st.image(overlay, use_container_width=True)

        # Zoom por articulación
        with st.expander("🔍 Zoom por articulación"):
            z_side = st.radio("Lado", ["Der", "Izq"], horizontal=True, key="zoom_side")
            z_joint = st.radio("Articulación", ["Cadera", "Rodilla", "Tobillo"], horizontal=True, key="zoom_joint")
            z_margin = st.slider("Margen (px)", 10, 200, 60, 10, key="zoom_margin")

            crop = get_joint_crop(img_path, payload, z_side, z_joint, z_margin)
            if crop:
                st.image(crop, caption=f"{z_side} · {z_joint}", use_container_width=True)
            else:
                st.info(f"No se detectó {z_joint} ({z_side})")

    # ── Panel derecho ───────────────────────────────────────────────────────
    with col_right:
        st.markdown("### 🎛️ Info")

        # Detecciones encontradas
        st.markdown("**Detecciones**")
        JOINTS = ["Cadera", "Rodilla", "Tobillo"]
        for side in ["Der", "Izq"]:
            found = payload.get("sides", {}).get(side, {}).get("detections_found", [])
            pills = ""
            for j in JOINTS:
                cls = "det-pill" if j in found else "det-pill det-missing"
                icon = "✓" if j in found else "✗"
                pills += f'<span class="{cls}">{icon} {j}</span>'
            st.markdown(
                f'<div style="margin-bottom:6px;"><b style="font-size:12px;">{side}</b><br>{pills}</div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # Descargas
        st.markdown("**Descargas**")

        json_bytes = json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")
        st.download_button(
            "📥 JSON métricas",
            data=json_bytes,
            file_name="cpak_metrics.json",
            mime="application/json",
            use_container_width=True,
        )

        img_bytes = BytesIO()
        overlay.save(img_bytes, format="JPEG", quality=92)
        st.download_button(
            "🖼️ Overlay JPG",
            data=img_bytes.getvalue(),
            file_name="cpak_overlay.jpg",
            mime="image/jpeg",
            use_container_width=True,
        )

        st.markdown("---")
        st.markdown("**JSON completo**")
        with st.expander("Ver payload"):
            st.json(payload, expanded=False)

else:
    # Estado inicial
    st.markdown(
        """
        <div style="text-align:center;padding:80px 0;color:#8b9ab1;">
            <div style="font-size:60px;">🦴</div>
            <div style="font-size:18px;margin-top:16px;">Carga una imagen Rx y presiona <b>Ejecutar inferencia</b></div>
            <div style="font-size:13px;margin-top:8px;">Usa el panel lateral para configurar los modelos y parámetros</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
