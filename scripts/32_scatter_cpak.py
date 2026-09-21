"""
Scatter plot JLO vs aHKA con cuadrícula CPAK (I-IX).
Lee los JSON del batch e incluye ambos lados (Der / Izq).
"""

import json
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ── Rutas ───────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[1]
BATCH_DIR  = ROOT / "outputs_data_json_batch_v2"
OUT_FILE   = ROOT / "reports" / "scatter_jlo_vs_ahka_v2.png"
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# ── Límites CPAK ────────────────────────────────────────────────────────────
AHKA_CUTS = (-2, 2)    # Varo < -2 | -2 <= Neutro <= 2 | Valgo > 2
JLO_CUTS  = (177, 183)  # Apex Distal < 177 | 177-183 Neutro | > 183 Apex Proximal

# Etiquetas para las 9 regiones (fila=aHKA, col=JLO)
# filas: 0=Varo(<-2), 1=Neutro, 2=Valgo(>2)
# cols: 0=Distal(<177), 1=Neutro, 2=Proximal(>183)
CPAK_LABEL = [
    ["I",  "IV", "VII"],
    ["II", "V",  "VIII"],
    ["III","VI", "IX"],
]

# ── Leer datos ───────────────────────────────────────────────────────────────
ahka_all, jlo_all, type_all = [], [], []

for p in sorted(BATCH_DIR.glob("*_final_metrics.json")):
    payload = json.loads(p.read_text(encoding="utf-8"))
    for side in ("Der", "Izq"):
        m = payload.get("sides", {}).get(side, {}).get("metrics", {})
        if m.get("status") == "ok":
            ahka_all.append(m["aHKA"])
            jlo_all.append(m["JLO"])
            type_all.append(m.get("CPAK_type", "?"))

ahka_all = np.array(ahka_all)
jlo_all  = np.array(jlo_all)

# Recortar a rango clínico razonable para mejor visualización
AHKA_RANGE = (-30, 30)
JLO_RANGE  = (155, 210)
mask = (
    (ahka_all >= AHKA_RANGE[0]) & (ahka_all <= AHKA_RANGE[1]) &
    (jlo_all  >= JLO_RANGE[0])  & (jlo_all  <= JLO_RANGE[1])
)
ahka_plot = ahka_all[mask]
jlo_plot  = jlo_all[mask]
type_plot = [t for t, m in zip(type_all, mask) if m]

n_total = len(ahka_all)
n_shown = mask.sum()
print(f"Puntos ok: {n_total}  |  mostrados en rango clínico: {n_shown}")

# ── Paleta de colores por tipo CPAK ─────────────────────────────────────────
CPAK_TYPES = ["I","II","III","IV","V","VI","VII","VIII","IX"]
cmap   = matplotlib.colormaps["tab10"]
COLORS = {t: cmap(i / 10) for i, t in enumerate(CPAK_TYPES)}
point_colors = [COLORS.get(t, "grey") for t in type_plot]

# Porcentaje de cada tipo respecto al total mostrado
type_counts = Counter(type_plot)
type_pct    = {t: type_counts[t] / n_shown * 100 for t in CPAK_TYPES}

# ── Figura ───────────────────────────────────────────────────────────────────
BG   = "white"
FG   = "#1a1a2e"
GRID = "#cccccc"

plt.rcParams.update({"font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(11, 8), dpi=140, facecolor=BG)
ax.set_facecolor(BG)

# Fondo de regiones alternado muy sutil
region_colors = ["#f7f9fc", "#eef1f7"]
ahka_regions = [AHKA_RANGE[0], AHKA_CUTS[0], AHKA_CUTS[1], AHKA_RANGE[1]]
jlo_regions  = [JLO_RANGE[0], JLO_CUTS[0], JLO_CUTS[1], JLO_RANGE[1]]

for row in range(3):
    for col in range(3):
        color = region_colors[(row + col) % 2]
        ax.fill_betweenx(
            [jlo_regions[col], jlo_regions[col+1]],
            ahka_regions[row], ahka_regions[row+1],
            color=color, zorder=0
        )

# Líneas de corte CPAK
line_kw = dict(color="#888aaa", lw=1.4, ls="--", zorder=1)
for v in AHKA_CUTS:
    ax.axvline(v, **line_kw)
for v in JLO_CUTS:
    ax.axhline(v, **line_kw)

# Etiquetas de región (tipo CPAK)
ahka_mids = [
    (AHKA_RANGE[0] + AHKA_CUTS[0]) / 2,
    (AHKA_CUTS[0]  + AHKA_CUTS[1]) / 2,
    (AHKA_CUTS[1]  + AHKA_RANGE[1]) / 2,
]
jlo_mids = [
    (JLO_RANGE[0] + JLO_CUTS[0]) / 2,
    (JLO_CUTS[0]  + JLO_CUTS[1]) / 2,
    (JLO_CUTS[1]  + JLO_RANGE[1]) / 2,
]
for row in range(3):
    for col in range(3):
        label = CPAK_LABEL[row][col]
        pct   = type_pct.get(label, 0)
        n_reg = type_counts.get(label, 0)
        # Número del tipo (grande, semitransparente)
        ax.text(
            ahka_mids[row], jlo_mids[col] + 0.6, label,
            ha="center", va="center",
            fontsize=28, fontweight="bold",
            color=COLORS[label], alpha=0.22,
            zorder=2
        )
        # Porcentaje (pequeño, debajo)
        ax.text(
            ahka_mids[row], jlo_mids[col] - 1.2,
            f"{pct:.1f}%  (n={n_reg})",
            ha="center", va="center",
            fontsize=7.5, color=COLORS[label], alpha=0.70,
            zorder=2
        )

# Scatter
sc = ax.scatter(
    ahka_plot, jlo_plot,
    c=point_colors, s=22, alpha=0.65,
    linewidths=0, zorder=3
)

# ── Estadísticas por tipo en el rango mostrado ───────────────────────────────
counts = type_counts

# ── Eje secundario derecho: etiquetas de columna JLO ────────────────────────
ax2 = ax.twinx()
ax2.set_ylim(ax.get_ylim())
ax2.set_yticks(jlo_mids)
ax2.set_yticklabels(["Apex\nDistal", "Neutro\nJLO", "Apex\nProximal"],
                    fontsize=8, color="#555577")
ax2.tick_params(axis="y", length=0, pad=6)
ax2.set_facecolor(BG)
for sp in ax2.spines.values():
    sp.set_visible(False)

# ── Eje superior: etiquetas de fila aHKA ────────────────────────────────────
ax3 = ax.twiny()
ax3.set_xlim(ax.get_xlim())
ax3.set_xticks(ahka_mids)
ax3.set_xticklabels(["Varo", "Neutro\naHKA", "Valgo"],
                    fontsize=8, color="#555577")
ax3.tick_params(axis="x", length=0, pad=6)
ax3.set_facecolor(BG)
for sp in ax3.spines.values():
    sp.set_visible(False)

# ── Decoración principal ─────────────────────────────────────────────────────
ax.set_xlabel("aHKA  (°)", color=FG, fontsize=12, labelpad=10)
ax.set_ylabel("JLO  (°)", color=FG, fontsize=12, labelpad=10)
ax.set_xlim(*AHKA_RANGE)
ax.set_ylim(*JLO_RANGE)
ax.tick_params(colors=FG, labelsize=9)
for sp in ax.spines.values():
    sp.set_color(GRID)
ax.xaxis.label.set_color(FG)
ax.yaxis.label.set_color(FG)
ax.grid(True, color="#e0e0e8", lw=0.5, ls=":", zorder=0)

# Líneas de referencia clínica (0°)
ax.axvline(0, color="#aaaacc", lw=0.9, ls="-", zorder=1, alpha=0.6)
ax.axhline(180, color="#aaaacc", lw=0.9, ls="-", zorder=1, alpha=0.6)

# ── Leyenda ──────────────────────────────────────────────────────────────────
legend_handles = [
    Line2D([0], [0], marker="o", color="none",
           markerfacecolor=COLORS[t], markersize=7,
           label=f"  Tipo {t}  ({counts.get(t, 0)})")
    for t in CPAK_TYPES
]
legend_handles += [
    Line2D([0], [0], color="#888aaa", lw=1.4, ls="--", label="Cortes CPAK"),
    Line2D([0], [0], color="#aaaacc", lw=0.9, ls="-",  label="Referencia 0° / 180°"),
]
leg = ax.legend(
    handles=legend_handles,
    loc="lower right", ncol=2,
    framealpha=0.9, facecolor="white",
    edgecolor="#ccccdd", labelcolor=FG,
    fontsize=8.5, title="CPAK",
    title_fontsize=9,
)
leg.get_title().set_color("#555577")

# Título
fig.text(
    0.5, 0.97,
    "Clasificación CPAK — JLO vs aHKA",
    ha="center", va="top",
    fontsize=15, fontweight="bold", color=FG,
)
fig.text(
    0.5, 0.935,
    f"n = {n_shown} rodillas  (de {n_total} con cálculo ok; outliers clínicos excluidos)",
    ha="center", va="top",
    fontsize=9, color="#8891b0",
)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(OUT_FILE, dpi=140, bbox_inches="tight", facecolor=BG)
print(f"Gráfico guardado en: {OUT_FILE}")
plt.show()
