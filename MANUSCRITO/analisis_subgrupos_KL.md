# Análisis por Subgrupos: ICC según Severidad de Artrosis (KL) y Prótesis

**Fecha:** 2026-07-16  
**Total:** 57 estudios, 112 observaciones (Der + Izq como independientes)  
**Clasificación:** Kellgren-Lawrence (KL) 0–4 por lado, Prótesis

---

## Distribución de la Muestra

| Grupo | N (lados) | % |
|-------|:---------:|:-:|
| KL 0 (normal) | 36 | 32.1% |
| KL 1 (mínima) | 9 | 8.0% |
| KL 2 (moderada) | 10 | 8.9% |
| KL 3 (severa) | 10 | 8.9% |
| KL 4 (muy severa) | 23 | 20.5% |
| Prótesis | 12 | 10.7% |
| *Sin clasificar* | *12* | *10.7%* |

---

## Resultados por Métrica

### HKA — Robusta en todos los subgrupos

| Grupo | N | ICC | MAE (°) | RMSE (°) | Bias (°) |
|-------|:-:|:---:|:-------:|:--------:|:--------:|
| KL 0 | 36 | **0.991** | 0.35 | 0.52 | −0.22 |
| KL 1 | 9 | 0.977 | 0.38 | 0.44 | +0.00 |
| KL 2 | 10 | 0.996 | 0.31 | 0.33 | −0.01 |
| KL 3 | 10 | 0.991 | 0.29 | 0.37 | +0.02 |
| KL 4 | 23 | **0.997** | 0.45 | 0.65 | −0.21 |
| Prótesis | 12 | 0.994 | 0.26 | 0.34 | +0.12 |

> El alineamiento global (HKA) se mantiene excelente independientemente de la severidad de la artrosis o presencia de prótesis (ICC > 0.97 en todos los grupos).

### mLDFA — Degradación significativa en artrosis severa (KL 4)

| Grupo | N | ICC | MAE (°) | RMSE (°) | Bias (°) |
|-------|:-:|:---:|:-------:|:--------:|:--------:|
| KL 0 | 36 | **0.816** | 0.75 | 0.98 | +0.34 |
| KL 1 | 9 | 0.955 | 0.77 | 0.96 | +0.56 |
| KL 2 | 10 | 0.952 | 0.60 | 0.71 | +0.05 |
| KL 3 | 10 | 0.877 | 0.87 | 1.03 | −0.59 |
| KL 4 | 23 | **0.589** | **1.85** | **2.09** | −0.81 |
| Prótesis | 12 | 0.918 | 0.72 | 0.93 | +0.65 |

> **Hallazgo principal:** El ICC de mLDFA cae de 0.816 en KL 0 a **0.589 en KL 4**, con un MAE que se duplica (0.75° → 1.85°). La deformidad del cóndilo femoral en artrosis avanzada distorsiona la detección del keypoint.

### mMPTA — MAE aumenta con la severidad

| Grupo | N | ICC | MAE (°) | RMSE (°) | Bias (°) |
|-------|:-:|:---:|:-------:|:--------:|:--------:|
| KL 0 | 36 | **0.842** | 1.26 | 1.60 | +0.55 |
| KL 1 | 9 | 0.928 | 0.61 | 0.73 | +0.35 |
| KL 2 | 10 | 0.829 | 1.12 | 1.22 | +0.30 |
| KL 3 | 10 | 0.877 | 1.01 | 1.21 | +0.99 |
| KL 4 | 23 | **0.869** | **2.19** | **2.71** | +1.11 |
| Prótesis | 12 | 0.899 | 0.74 | 0.99 | +0.15 |

> El MAE de mMPTA casi se duplica en KL 4 (2.19° vs 1.26° en KL 0), aunque el ICC se mantiene relativamente estable (~0.87).

---

## Resumen de Tendencias

| Métrica | Tendencia con severidad (KL 0 → 4) | Grupo más afectado |
|---------|:-----------------------------------:|:------------------:|
| **HKA** | Sin degradación (ICC ~0.99) | — |
| **mLDFA** | **Degradación severa** (0.82 → 0.59) | KL 4 (MAE 1.85°) |
| **mMPTA** | MAE aumenta (1.26° → 2.19°) | KL 4 (MAE 2.19°) |

### Interpretación Clínica

1. **mLDFA en KL 4** es el punto débil del modelo. La pérdida de la morfología normal del cóndilo femoral en artrosis avanzada (KL 4) probablemente confunde al detector de keypoints, resultando en una medición menos fiable del ángulo femoral distal.

2. **HKA se mantiene robusto** porque combina información de cadera, rodilla y tobillo, promediando errores locales.

3. **Prótesis** no parece afectar significativamente la precisión del modelo (ICC > 0.90 en todas las métricas), posiblemente porque los bordes metálicos de la prótesis son más fáciles de detectar que el hueso deformado.

---

*Nota: Los grupos KL 1, 2 y 3 tienen N pequeño (9–10 lados), por lo que sus estimaciones de ICC deben interpretarse con precaución.*
