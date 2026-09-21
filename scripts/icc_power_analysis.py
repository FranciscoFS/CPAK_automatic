"""
Análisis de poder estadístico para ICC — Verificación completa.

Métodos:
  1. Walter, Eliasziw & Donner (1998): test de hipótesis H0: ICC <= rho0 vs H1: ICC > rho0.
     - Aproximación z de Fisher (como en icc_sample_size.py)
     - Método EXACTO con distribución F (F0*F_{n-1,n(k-1)} bajo H0)
  2. Bonett (2002): N según precisión del IC.
  3. Poder ALCANZADO con el n real de cada métrica y los ICC observados.
  4. Verificación de la afirmación: "con 2 evaluadores, <30 imágenes es suficiente".

Datos: reports/validation_round/Mediciones/mediciones_{IA,FFS,JS}.xlsx
"""

import math
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import pingouin as pg

warnings.filterwarnings("ignore")

BASE = r"reports/validation_round/Mediciones"
METRICS = ["HKA", "mLDFA", "mMPTA"]


# ─────────────────────────────────────────────────────────────
# 1. Cargar datos reales y n por métrica
# ─────────────────────────────────────────────────────────────
def load_data():
    """Carga las mediciones en formato largo: una fila por RUT+ID+Lado."""
    def load_long(path):
        df = pd.read_excel(path)
        df = df.copy()
        df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]
        # Pivotar Der/Izq a formato largo
        parts = []
        for lado in ["Der", "Izq"]:
            sub = df[["RUT", "ID_estudio"] + [f"{m}_{lado}" for m in METRICS]].copy()
            sub["Lado"] = lado
            sub = sub.rename(columns={f"{m}_{lado}": m for m in METRICS})
            parts.append(sub)
        long = pd.concat(parts, ignore_index=True)
        long["_key"] = long["RUT"].astype(str) + "_" + long["ID_estudio"].astype(str) + "_" + long["Lado"]
        long = long.drop_duplicates("_key").set_index("_key")
        return long

    ia = load_long(f"{BASE}/mediciones_IA.xlsx")
    ffs = load_long(f"{BASE}/mediciones_FFS.xlsx")
    js = load_long(f"{BASE}/mediciones_JS.xlsx")

    out = {}
    for m in METRICS:
        sub = pd.DataFrame({"IA": ia[m], "FFS": ffs[m], "JS": js[m]}).dropna()
        out[m] = sub
    return out


# ─────────────────────────────────────────────────────────────
# 2. Funciones de poder / tamaño muestral
# ─────────────────────────────────────────────────────────────
def f_icc(rho, k):
    """F esperado para un ICC dado con k evaluadores (one-way)."""
    return (1 + (k - 1) * rho) / (1 - rho)


def power_exact(n, k, rho0, rho1, alpha=0.05):
    """
    Poder EXACTO del test H0: rho=rho0 vs H1: rho=rho1 (unilateral).
    Bajo H0: F ~ F0 * F_{n-1, n(k-1)}; se rechaza si F > F0 * Fcrit.
    """
    if rho1 <= rho0:
        return np.nan
    F0 = f_icc(rho0, k)
    F1 = f_icc(rho1, k)
    df1, df2 = n - 1, n * (k - 1)
    if df1 < 1 or df2 < 1:
        return np.nan
    fcrit = stats.f.ppf(1 - alpha, df1, df2)
    # Poder = P(F1*X > F0*fcrit) = P(X > (F0/F1)*fcrit)
    return 1 - stats.f.cdf((F0 / F1) * fcrit, df1, df2)


def n_needed_exact(k, rho0, rho1, alpha=0.05, power=0.80, n_max=5000):
    """Menor n con poder >= power (método exacto)."""
    for n in range(3, n_max + 1):
        if power_exact(n, k, rho0, rho1, alpha) >= power:
            return n
    return n_max


def power_fisher(n, k, rho0, rho1, alpha=0.05):
    """Aproximación z de Fisher (como en icc_sample_size.py)."""
    z0 = 0.5 * math.log((1 + (k - 1) * rho0) / (1 - rho0))
    z1 = 0.5 * math.log((1 + (k - 1) * rho1) / (1 - rho1))
    z_alpha = stats.norm.ppf(1 - alpha)
    z_obs = (z1 - z0) * math.sqrt(n - 2)
    return stats.norm.cdf(z_obs - z_alpha)


def n_needed_fisher(k, rho0, rho1, alpha=0.05, power=0.80):
    z_alpha = stats.norm.ppf(1 - alpha)
    z_beta = stats.norm.ppf(power)
    z0 = 0.5 * math.log((1 + (k - 1) * rho0) / (1 - rho0))
    z1 = 0.5 * math.log((1 + (k - 1) * rho1) / (1 - rho1))
    return math.ceil(((z_alpha + z_beta) / (z1 - z0)) ** 2 + 2)


def ci_width_at_n(n, k, rho_est, alpha=0.05):
    """Semiancho del IC95% del ICC estimado (approx. Bonett/Fisher) para n dado."""
    z_alpha2 = stats.norm.ppf(1 - alpha / 2)
    var_z = (2 * (k - 1)) / (k * (k - 1))
    z_r = 0.5 * math.log((1 + (k - 1) * rho_est) / (1 - rho_est))
    # error estándar en escala z
    se_z = math.sqrt(var_z / (n - 2)) if n > 2 else float("inf")
    half_z = z_alpha2 * se_z
    # volver a escala ICC
    def z_to_icc(z):
        return (math.exp(2 * z) - 1) / (math.exp(2 * z) + k - 1)
    lo = max(z_to_icc(z_r - half_z), -0.999)
    hi = min(z_to_icc(z_r + half_z), 0.999)
    return (hi - lo) / 2, lo, hi


# ─────────────────────────────────────────────────────────────
# 3. Análisis
# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 78)
    print("  ANÁLISIS DE PODER ESTADÍSTICO PARA ICC")
    print("  Walter, Eliasziw & Donner (1998) + Bonett (2002)")
    print("=" * 78)

    # ---- A) Verificar afirmación: 2 raters, <30 imágenes ----
    print("\n" + "-" * 78)
    print("  A) ¿ES SUFICIENTE CON <30 IMÁGENES Y 2 EVALUADORES?")
    print("  (α=0.05 unilateral, potencia 80% y 90%)")
    print("-" * 78)
    print(f"  {'ρ0':<6}{'ρ1':<6}{'k':<4}{'N (Fisher)':<12}{'N (Exacto)':<12}{'N (Fisher)':<12}")
    print(f"  {'':<6}{'':<6}{'':<4}{'80% pow':<12}{'80% pow':<12}{'90% pow':<12}")
    print(f"  {'-'*5}{'-'*6}{'-'*4}{'-'*12}{'-'*12}{'-'*12}")
    scenarios = [
        (0.70, 0.90),  # clásico: mínimo aceptable 0.70, esperado 0.90
        (0.75, 0.90),  # mínimo 0.75, esperado 0.90
        (0.70, 0.85),  # más exigente
        (0.60, 0.90),  # menos exigente
        (0.80, 0.95),  # concordancia excelente
    ]
    for rho0, rho1 in scenarios:
        nf80 = n_needed_fisher(2, rho0, rho1, power=0.80)
        ne80 = n_needed_exact(2, rho0, rho1, power=0.80)
        nf90 = n_needed_fisher(2, rho0, rho1, power=0.90)
        print(f"  {rho0:<6.2f}{rho1:<6.2f}{2:<4}{nf80:<12}{ne80:<12}{nf90:<12}")

    # ---- B) Comparación Fisher vs Exacto ----
    print("\n" + "-" * 78)
    print("  B) VALIDACIÓN: Fisher (aprox.) vs F exacta")
    print("-" * 78)
    print(f"  {'n':<6}{'k':<4}{'ρ0':<6}{'ρ1':<6}{'Power Fisher':<14}{'Power Exacto':<14}")
    for n, k in [(19, 2), (30, 2), (112, 2), (112, 3), (50, 3), (100, 3)]:
        for rho0, rho1 in [(0.70, 0.90), (0.70, 0.85), (0.60, 0.80)]:
            pf = power_fisher(n, k, rho0, rho1)
            pe = power_exact(n, k, rho0, rho1)
            print(f"  {n:<6}{k:<4}{rho0:<6.2f}{rho1:<6.2f}{pf:<14.3f}{pe:<14.3f}")

    # ---- C) Datos reales: n, ICC observado, poder alcanzado ----
    print("\n" + "-" * 78)
    print("  C) DATOS REALES: n, ICC observado y PODER ALCANZADO")
    print("-" * 78)
    data = load_data()

    for m in METRICS:
        sub = data[m]
        n = len(sub)
        print(f"\n  ── {m}: n = {n} ──")

        # ICC multi-observador (3 raters: IA+FFS+JS)
        long = pd.DataFrame({
            "id": list(sub.index) * 3,
            "Obs": ["IA"] * n + ["FFS"] * n + ["JS"] * n,
            "val": pd.concat([sub["IA"], sub["FFS"], sub["JS"]]).values,
        })
        icc_df = pg.intraclass_corr(data=long, targets="id", raters="Obs", ratings="val")
        row = icc_df[icc_df["Type"] == "ICC(A,1)"].iloc[0]
        icc3 = row["ICC"]
        ci3 = row["CI95"]
        print(f"    ICC(A,1) 3 observadores = {icc3:.4f}  IC95% [{ci3[0]:.4f}, {ci3[1]:.4f}]")

        # Poder alcanzado para detectar ICC>0.7 y ICC>0.75 (k=3)
        for rho0 in [0.60, 0.70, 0.75]:
            p = power_exact(n, 3, rho0, icc3)
            print(f"    Poder (H1: ρ={icc3:.3f} vs ρ0={rho0}) = {p*100:.1f}%")

        # ICC pairwise IA vs FFS (2 raters)
        sub2 = sub.dropna(subset=["IA", "FFS"])
        n2 = len(sub2)
        long2 = pd.DataFrame({
            "id": list(sub2.index) * 2,
            "Obs": ["IA"] * n2 + ["FFS"] * n2,
            "val": pd.concat([sub2["IA"], sub2["FFS"]]).values,
        })
        icc_df2 = pg.intraclass_corr(data=long2, targets="id", raters="Obs", ratings="val")
        row2 = icc_df2[icc_df2["Type"] == "ICC(A,1)"].iloc[0]
        icc2 = row2["ICC"]
        ci2 = row2["CI95"]
        print(f"    ICC(A,1) IA vs FFS = {icc2:.4f}  IC95% [{ci2[0]:.4f}, {ci2[1]:.4f}]  (n={n2})")
        for rho0 in [0.60, 0.70, 0.75]:
            p = power_exact(n2, 2, rho0, icc2)
            print(f"    Poder (H1: ρ={icc2:.3f} vs ρ0={rho0}) = {p*100:.1f}%")

        # Precisión del IC con n real
        w, lo, hi = ci_width_at_n(n, 3, icc3)
        print(f"    Precisión IC95% (k=3, ρ≈{icc3:.3f}): semiancho ≈ ±{w:.3f}  → IC [{lo:.3f}, {hi:.3f}]")

    # ---- D) Resumen de n real vs N requerido ----
    print("\n" + "-" * 78)
    print("  D) RESUMEN: n real (112) vs N requerido (k=3, 80%)")
    print("-" * 78)
    print(f"  {'ρ0':<6}{'ρ1':<6}{'N req (Fisher)':<16}{'N req (Exacto)':<16}")
    for rho0, rho1 in [(0.70, 0.90), (0.70, 0.85), (0.60, 0.85), (0.75, 0.90)]:
        nf = n_needed_fisher(3, rho0, rho1)
        ne = n_needed_exact(3, rho0, rho1)
        print(f"  {rho0:<6.2f}{rho1:<6.2f}{nf:<16}{ne:<16}")

    print("\n  NOTA: el poder 'alcanzado' usa el ICC observado como H1; es un")
    print("  análisis post-hoc. El cálculo a priori (A) es el recomendable para el paper.")


if __name__ == "__main__":
    main()
