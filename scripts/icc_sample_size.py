"""
Cálculo del tamaño muestral (N) necesario para estudios de ICC interobservador.

Basado en:
  - Walter, Eliasziw, Donner (1998) "Sample size and optimal designs for reliability studies"
    Statistics in Medicine, 17(1), 101-110.
  - Bonett, D.G. (2002) "Sample size requirements for estimating intraclass correlations
    with desired precision" Statistics in Medicine, 21(9), 1331-1335.

Utiliza la transformación z de Fisher del ICC para k evaluadores:
  z(ρ) = 0.5 * ln((1 + (k-1)ρ) / (1-ρ))
"""

import math
import argparse
from scipy import stats


def icc_to_z(rho, k):
    """Transformación z de Fisher del ICC para k evaluadores."""
    return 0.5 * math.log((1 + (k - 1) * rho) / (1 - rho))


def sample_size_hypothesis_test(k, rho0, rho1, alpha=0.05, power=0.80, alternative="greater"):
    """
    Calcula N necesario para test de hipótesis H0: ICC ≤ ρ0 vs H1: ICC > ρ0.
    
    Parámetros:
        k : int — número de evaluadores/observadores
        rho0 : float — ICC mínimo aceptable bajo H0
        rho1 : float — ICC esperado bajo H1
        alpha : float — nivel de significancia (default 0.05)
        power : float — potencia deseada (default 0.80)
        alternative : str — "greater" (unilateral) o "two-sided" (bilateral)
    
    Returns:
        n : int — sujetos necesarios
    """
    z_alpha = stats.norm.ppf(1 - alpha) if alternative == "greater" else stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    
    z0 = icc_to_z(rho0, k)
    z1 = icc_to_z(rho1, k)
    
    n = ((z_alpha + z_beta) / (z1 - z0)) ** 2 + 2
    return math.ceil(n)


def sample_size_ci_width(k, rho_expected, w, alpha=0.05):
    """
    Calcula N necesario para lograr un ancho de intervalo de confianza deseado.
    Método de Bonett (2002) aproximado.
    
    Parámetros:
        k : int — número de evaluadores
        rho_expected : float — ICC que se espera observar
        w : float — semiancho del IC deseado (p.ej., 0.10 = ±0.10)
        alpha : float — nivel de significancia
    
    Returns:
        n : int — sujetos necesarios
    """
    z_alpha2 = stats.norm.ppf(1 - alpha / 2)
    
    # Varianza aproximada del z-transformado
    var_z = (2 * (k - 1)) / (k * (k - 1))
    
    z_r = icc_to_z(rho_expected, k)
    
    # n = (z_alpha/2)^2 * var_z / (semiancho en escala z)^2
    # El semiancho w está en escala ICC, lo aproximamos en escala z
    z_upper = icc_to_z(min(rho_expected + w, 0.999), k)
    z_lower = icc_to_z(max(rho_expected - w, -0.999), k)
    z_w = (z_upper - z_lower) / 2
    
    if z_w <= 0:
        return float('inf')
    
    n = (z_alpha2 / z_w) ** 2 * var_z + 2
    return math.ceil(n)


def tabelar_n(k=2, alpha=0.05, power=0.80):
    """Genera tabla de N para distintos escenarios típicos."""
    print(f"\n{'='*80}")
    print(f"  TAMAÑO MUESTRAL (N) PARA ICC INTEROBSERVADOR")
    print(f"  {k} evaluadores | α = {alpha} | potencia = {1-power:.0%} → {power:.0%}")
    print(f"{'='*80}")
    print(f"\n  Test unilateral H₀: ICC ≤ ρ₀  vs  H₁: ICC > ρ₀\n")
    print(f"  {'ρ₀ (mín. aceptable)':<22} {'ρ₁ (ICC esperado)':<22} {'N necesario':<12}")
    print(f"  {'-'*22} {'-'*22} {'-'*12}")
    
    escenarios_rho0 = [0.40, 0.50, 0.60, 0.70, 0.75]
    escenarios_rho1 = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    
    for rho0 in escenarios_rho0:
        for rho1 in escenarios_rho1:
            if rho1 <= rho0:
                continue
            n = sample_size_hypothesis_test(k, rho0, rho1, alpha, power)
            print(f"  {rho0:<22.2f} {rho1:<22.2f} {n:<12}")
    
    print(f"\n  {'='*80}")
    print(f"  SEGÚN PRECISIÓN DEL IC (Bonett, 2002)")
    print(f"  {'='*80}")
    print(f"\n  {'ICC esperado':<16} {'Semiancho IC':<16} {'N necesario':<12}")
    print(f"  {'-'*16} {'-'*16} {'-'*12}")
    for rho_exp in [0.80, 0.85, 0.90, 0.95]:
        for w in [0.05, 0.08, 0.10, 0.15]:
            n = sample_size_ci_width(k, rho_exp, w, alpha)
            print(f"  {rho_exp:<16.2f} ±{w:<14.2f} {n:<12}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cálculo de N para ICC interobservador")
    parser.add_argument("--k", type=int, default=2, help="Número de evaluadores (default: 2)")
    parser.add_argument("--alpha", type=float, default=0.05, help="Nivel de significancia (default: 0.05)")
    parser.add_argument("--power", type=float, default=0.80, help="Potencia (default: 0.80)")
    parser.add_argument("--rho0", type=float, help="ICC mínimo aceptable (H0)")
    parser.add_argument("--rho1", type=float, help="ICC esperado (H1)")
    parser.add_argument("--rho-expected", type=float, help="ICC esperado (para método CI width)")
    parser.add_argument("--ci-width", type=float, help="Semiancho del IC deseado")
    args = parser.parse_args()
    
    if args.rho0 is not None and args.rho1 is not None:
        n = sample_size_hypothesis_test(args.k, args.rho0, args.rho1, args.alpha, args.power)
        print(f"\n  N necesario = {n} sujetos")
        print(f"  (k={args.k}, ρ₀={args.rho0}, ρ₁={args.rho1}, α={args.alpha}, potencia={args.power:.0%})")
    
    if args.rho_expected is not None and args.ci_width is not None:
        n = sample_size_ci_width(args.k, args.rho_expected, args.ci_width, args.alpha)
        print(f"\n  N necesario (IC) = {n} sujetos")
        print(f"  (k={args.k}, ICC esperado={args.rho_expected}, semiancho IC=±{args.ci_width}, α={args.alpha})")
    
    if args.rho0 is None and args.rho1 is None and args.rho_expected is None:
        # Modo tabla
        tabelar_n(args.k, args.alpha, args.power)
