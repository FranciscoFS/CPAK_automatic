# Resultados ICC — Análisis Multi-Observador

**Fecha:** 2026-07-21  
**Pares:** 112 (57 estudios × 2 lados − 2 lados Izq sin medición FFS)  
**Observadores:** IA (YOLO26s + Pose v2 FT), FFS (Francisco), JS

---

## 1. ICC (A,1) — 3 observadores simultáneos

| Métrica | ICC | IC95% | Clasificación |
|---------|:---:|:-----|:-------------|
| **HKA** | **0.995** | 0.99–1.00 | Excelente |
| **mLDFA** | **0.873** | 0.83–0.91 | Buena |
| **mMPTA** | **0.873** | 0.80–0.92 | Buena |

## 2. ICC pairwise IA vs cada observador

| Par | HKA | mLDFA | mMPTA |
|:---|:---:|:-----:|:-----:|
| IA vs FFS | 0.995 | 0.864 | 0.877 |
| IA vs JS | 0.994 | 0.834 | 0.827 |
| FFS vs JS | — | — | — |

## 3. CCC (Lin) — IA vs FFS como gold standard

| Métrica | CCC | Pearson r | Error sistemático |
|:-------|:---:|:---------:|:----------------:|
| **HKA** | **0.995** | 0.995 | +0.11° |
| **mLDFA** | **0.863** | 0.869 | +0.02° |
| **mMPTA** | **0.876** | 0.914 | −0.64° |

## 4. Error sistemático (bias) completo

| Par | HKA | mLDFA | mMPTA |
|:---|:---:|:-----:|:-----:|
| IA vs FFS | +0.11° (LoA: −0.86 a +1.08) | +0.02° (LoA: −2.50 a +2.55) | −0.65° (LoA: −3.77 a +2.48) |
| IA vs JS | +0.29° (LoA: −0.68 a +1.27) | −0.33° (LoA: −2.98 a +2.32) | −0.98° (LoA: −4.37 a +2.40) |
| FFS vs JS | +0.18° (LoA: −0.55 a +0.91) | −0.36° (LoA: −1.87 a +1.15) | −0.34° (LoA: −2.39 a +1.72) |
| **IA vs promedio obs.** | **+0.20° (LoA: −0.70 a +1.10)** | **−0.15° (LoA: −2.63 a +2.32)** | **−0.81° (LoA: −3.91 a +2.28)** |

## 5. Kappa — Clasificaciones CPAK

### aHKA class (Varo/Neutro/Valgo)

| Par | κ ponderado | κ simple |
|:---|:-----------:|:--------:|
| IA vs FFS | 0.545 | 0.567 |
| IA vs JS | 0.508 | 0.579 |
| FFS vs JS | **0.643** | 0.677 |
| **Fleiss (3 obs)** | **—** | **0.606** |

### JLO class (Distal/Neutro/Proximal)

| Par | κ ponderado | κ simple |
|:---|:-----------:|:--------:|
| IA vs FFS | 0.673 | 0.656 |
| IA vs JS | 0.639 | 0.640 |
| FFS vs JS | **0.786** | 0.778 |
| **Fleiss (3 obs)** | **—** | **0.692** |

### CPAK type (I–IX)

| Par | κ simple | Acuerdo (%) |
|:---|:--------:|:-----------:|
| IA vs FFS | 0.525 | 61.6% |
| IA vs JS | 0.573 | 65.2% |
| FFS vs JS | **0.646** | 71.4% |
| **Fleiss (3 obs)** | **0.580** | — |

## 6. Heteroscedasticidad (Bland-Altman)

| Métrica | r(|diff|, media) | p-valor | ¿Heteroscedástico? |
|:-------|:---------------:|:-------:|:------------------:|
| HKA | +0.004 | 0.97 | No |
| mLDFA | −0.056 | 0.56 | No |
| mMPTA | **−0.279** | **0.003** | **Sí ⚠** |

> El mMPTA presenta mayor error en ángulos bajos (< 86°): |diff| medio = 1.84° en el cuartil inferior vs 0.96° en el superior.

---

## 7. Poder estadístico y tamaño muestral

**Cálculo a priori — Walter, Eliasziw & Donner (1998), test unilateral H₀: ICC ≤ ρ₀ vs H₁: ICC > ρ₀ (α=0.05).**

| k (evaluadores) | ρ₀ (mín. aceptable) | ρ₁ (ICC esperado) | N (80% poder) | N (90% poder) |
|:---:|:---:|:---:|:---:|:---:|
| 2 | 0.70 | 0.90 | **19** | 26 |
| 2 | 0.75 | 0.90 | **27** | 37 |
| 2 | 0.70 | 0.85 | 43 | 59 |
| 2 | 0.80 | 0.95 | **14** | 18 |
| 3 | 0.70 | 0.90 | **13** | — |

→ Con 2 evaluadores y concordancia excelente esperada (ρ₁=0.90), **<30 sujetos son suficientes** (N=19–27).

**Poder alcanzado con el n real = 112 (por métrica):**

| Métrica | ICC (3 obs.) | Poder vs ρ₀=0.70 | Poder vs ρ₀=0.75 | Semiancho IC95% |
|:---|:---:|:---:|:---:|:---:|
| **HKA** | 0.995 | 100% | 100% | ±0.002 |
| **mLDFA** | 0.873 | 100% | 99.9% | ±0.036 |
| **mMPTA** | 0.873 | 100% | 99.8% | ±0.036 |

> El n=112 excede ampliamente el mínimo requerido (13–27), otorgando poder >99% incluso contra el umbral más exigente (ρ₀=0.75) y CIs muy estrechos. La aproximación z de Fisher difiere <2% del método exacto con distribución F (validación cruzada).

**Referencias:**
- Walter SD, Eliasziw M, Donner A. Sample size and optimal designs for reliability studies. *Stat Med.* 1998;17(1):101-110.
- Bonett DG. Sample size requirements for estimating intraclass correlations with desired precision. *Stat Med.* 2002;21(9):1331-1335.

---

## Archivos generados

- `visualizations_abstract/abstract_ia_vs_humanos.png` — Scatter + Bland-Altman IA vs promedio observadores
- `visualizations_abstract/02_icc_scatter_blandaltman.png` — Scatter + Bland-Altman FFS vs IA
- `reports/validation_round/Mediciones/` — Datos fuente de cada observador

---

*Análisis completo al 2026-07-21. Pendiente: incorporar mediciones de PB y CR.*
