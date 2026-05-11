# Informe de Resultados - Pipeline CPAK

**Fecha:** 2026-05-10  
**Proyecto:** CPAK (inferencia batch sobre carpeta `data/`)  
**Salida batch:** `visualizations/final_pipeline_data_batch/`

## 1) Resumen Ejecutivo

Se procesaron **597 estudios** y se obtuvieron métricas válidas en ambos lados en **524 casos (87.8%)**.

- **Cálculo completo (Der + Izq):** 524/597 (**87.8%**)
- **Cálculo parcial (1 lado):** 60/597 (**10.1%**)
- **Sin cálculo (0 lados):** 13/597 (**2.2%**)

En total se evaluaron **1194 lados** (2 por estudio):

- **Lados con estado `ok`:** 1108 (**92.8%**)
- **Lados `incomplete`:** 86 (**7.2%**)

## 2) Resultados Cuantitativos

### 2.1 Completitud por lado

- **Derecho (`Der`)**: 550/597 `ok` (**92.1%**), 47 incompletos
- **Izquierdo (`Izq`)**: 558/597 `ok` (**93.5%**), 39 incompletos

### 2.2 Distribuciones métricas (solo casos `ok`)

#### Lado Derecho (n=550)

- **LDFA:** media 87.71, mediana 87.07, std 5.90, rango [67.83, 163.94]
- **MPTA:** media 87.14, mediana 87.99, std 6.92, rango [31.17, 120.57]
- **aHKA:** media -0.57, mediana 0.90, std 8.90, rango [-66.83, 24.05]
- **JLO:** media 174.84, mediana 174.98, std 9.28, rango [129.18, 284.51]

#### Lado Izquierdo (n=558)

- **LDFA:** media 87.93, mediana 86.78, std 7.44, rango [19.32, 159.55]
- **MPTA:** media 85.55, mediana 87.04, std 8.09, rango [4.90, 112.41]
- **aHKA:** media -2.38, mediana -0.11, std 10.46, rango [-59.77, 37.71]
- **JLO:** media 173.47, mediana 173.42, std 11.49, rango [24.22, 271.96]

> Nota: Los rangos extremos sugieren un subconjunto con geometría atípica o inferencia ruidosa.

### 2.3 Distribución CPAK (tipos I-IX)

#### Derecho

- III: 144
- II: 131
- I: 93
- VI: 50
- V: 47
- VII: 29
- IV: 22
- IX: 18
- VIII: 16

#### Izquierdo

- II: 156
- I: 137
- III: 130
- VII: 32
- V: 27
- VI: 26
- IV: 21
- VIII: 15
- IX: 14

Patrón global: predominio de tipos **I-II-III**.

## 3) Revisión de Casos Incompletos

### 3.1 Impacto por estudio

- **Estudios afectados:** 73/597 (**12.2%**)
- **Con ambos lados incompletos:** 13
- **Con un solo lado incompleto:** 60

### 3.2 Principales articulaciones faltantes (detección)

- `Izq:Tobillo`: 25
- `Der:Rodilla`: 21
- `Der:Tobillo`: 12
- `Der:Cadera`: 5
- `Izq:Rodilla`: 3
- `Izq:Cadera`: 1

### 3.3 Patrones de keypoints faltantes más frecuentes

- `[1,3,4,5,6,7]`: 34 casos
- `[2]`: 34 casos
- `[1,2,3,4,5,6,7]`: 7 casos
- `[0]`: 6 casos
- `[0,1,3,4,5,6,7]`: 4 casos
- `[0,1,2,3,4,5,6,7]`: 1 caso

Interpretación inicial:

- `[2]` apunta a pérdida aislada del centro de tobillo.
- `[1,3,4,5,6,7]` sugiere fallo amplio alrededor de rodilla/fémur distal.

## 4) Visualización JLO vs aHKA

Se generó scatter con cuadrícula CPAK y cortes:

- **aHKA:** -2 y +2
- **JLO:** 177 y 183

Características del gráfico:

- Fondo blanco (estilo sobrio/científico)
- Cuadrícula CPAK visible
- Etiqueta por celda con **% y n**
- Rango de visualización clínica: aHKA [-30, 30], JLO [155, 210]

## 5) Puntos Prioritarios a Revisar

1. **Robustez de detección en tobillo**
   - Principal cuello de botella en lado izquierdo.
   - Revisar ejemplos con variaciones de exposición, recorte y contraste en región distal.

2. **Robustez de detección/pose en rodilla derecha**
   - Alto peso en incompletitud por falta de articulación o keypoints múltiples.
   - Inspeccionar fallos por superposición, rotación y artefactos.

3. **Control de calidad de outliers geométricos**
   - Implementar validaciones blandas para marcar casos extremos sin descartarlos automáticamente.
   - Propuesta de bandera `low_confidence_geometry` cuando métricas quedan fuera de rango clínico esperable.

4. **Análisis estratificado por condición de adquisición**
   - Separar fallos por tipo de imagen (centrado, magnificación, recorte de tobillo/cadera).
   - Esto permite priorizar mejoras de datos y no solo del modelo.

5. **Ciclo de mejora sugerido**
   - Curar 100-150 casos fallidos (especialmente tobillo izq y rodilla der)
   - Reentrenar detector + ajuste fino de pose
   - Reevaluar con misma batería batch y comparar `% completos` y dispersión de outliers

## 6) Archivos Relacionados

- Resumen batch en consola: `scripts/31_batch_summary.py`
- Revisión incompletos (script): `scripts/33_review_incomplete.py`
- CSV detallado incompletos: `reports/incomplete_cases_review.csv`
- Scatter CPAK: `reports/scatter_jlo_vs_ahka.png`

---

### Conclusión

El pipeline ya tiene un rendimiento operativo alto (**87.8% completos en ambos lados**), con una tasa de lados válidos de **92.8%**. El trabajo de mejora debería enfocarse en detección/pose de **tobillo (especialmente izquierdo)** y **rodilla derecha**, y en mecanismos de control para outliers geométricos.
