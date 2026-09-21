# CPAK — Pipeline de Entrenamiento

## Visión General

El sistema CPAK consta de **dos fases secuenciales** de Deep Learning:

1. **Detección de Bounding Boxes (Fase 1):** Localiza Cadera, Rodilla y Tobillo (izquierdo/derecho) en radiografías TeleRx.
2. **Pose / Keypoint Detection (Fase 2):** Ubica puntos anatómicos sobre los recortes (crops) generados en la Fase 1 para calcular las métricas CPAK (aHKA y JLO).

---

## Fase 1 — Detección de Articulaciones (Object Detection)

### Dataset

| Split | Imágenes |
|-------|----------|
| Train | 197 |
| Val   | 55 |
| Test  | 28 |
| **Total** | **280** |

**Clases:** 6 — Cadera Der, Cadera Izq, Rodilla Der, Rodilla Izq, Tobillo Der, Tobillo Izq.

### Modelos Entrenados

#### 1. `telerx_yolov8n` (YOLOv8 nano) — Histórico

| Parámetro | Valor |
|-----------|-------|
| Modelo base | `yolov8n.pt` |
| Épocas | 100 |
| Tamaño de imagen | 640 px |
| Batch size | 16 |
| mAP50 | **0.9695** |
| mAP50-95 | **0.6629** |

#### 2. `telerx_yolo26s_768_b12-4` (YOLO26s) — ✅ En producción

| Parámetro | Valor |
|-----------|-------|
| Modelo base | `yolo26s.pt` |
| Épocas | 100 |
| Tamaño de imagen | 768 px |
| Batch size | 12 |
| mAP50 | **0.9734** |
| mAP50-95 | **0.6774** |

> **Mejora respecto a YOLOv8n:** +0.4% mAP50, +1.5% mAP50-95. Se adoptó por su mayor resolución (768 vs 640) y mejor precisión general.

---

## Fase 2 — Detección de Puntos Anatómicos (Pose / Keypoints)

### Esqueleto Maestro

| ID | Landmark Anatómico | Crop Destino |
|----|-------------------|--------------|
| 0  | Centro Cabeza Femoral | Cadera |
| 1  | Centro Espinas Tibiales | Rodilla |
| 2  | Centro Domo del Talo | Tobillo |
| 3  | Cóndilo Medial (Vértice) | Rodilla |
| 4  | Cóndilo Lateral (Vértice) | Rodilla |
| 5  | Plateau Medial (Centro) | Rodilla |
| 6  | Plateau Lateral (Centro) | Rodilla |
| 7  | Punto adicional (v2) | — |

### Dataset Progresivo

| Versión | Train | Val | Test | **Total** |
|---------|-------|-----|------|-----------|
| `dataset_pose_final` | ~850 | ~93 | — | **~943** |
| `dataset_pose_v2` | 1.387 | 268 | 79 | **1.734** |

### Modelos Entrenados

#### 1. `telerx_pose_s_rebuild1_ft_full_b16_flipfix` — Base

| Parámetro | Valor |
|-----------|-------|
| Dataset | `dataset_pose_final` (~943 imgs) |
| Épocas | 30 |
| Tamaño de imagen | 800 px |
| Batch size | 16 |
| Pose mAP50 | **0.9774** |
| Pose mAP50-95 | **0.9703** |
| Box mAP50 | **0.9950** |

#### 2. `telerx_pose_v2_ft` — ✅ En producción (fine-tune)

| Parámetro | Valor |
|-----------|-------|
| Dataset | `dataset_pose_v2` (1.734 imgs) |
| Pesos iniciales | `best.pt` del modelo base |
| Épocas | 40 |
| Tamaño de imagen | 800 px |
| Batch size | 16 |
| Pose mAP50 | **0.9831** |
| Pose mAP50-95 | **0.9809** |
| Box mAP50 | **0.9948** |
| Keypoint shape | [8, 3] |

> **Mejora respecto al modelo base:** +0.6% mAP50, +1.1% mAP50-95 al incorporar ~790 nuevos crops etiquetados.

---

## Pipeline de Inferencia

Los modelos activos en producción son:

```
TeleRx (radiografía completa)
    │
    ▼
┌──────────────────────────────────────┐
│ Detección: YOLO26s (768 px)          │
│ telerx_yolo26s_768_b12-4/best.pt     │
│ → Bounding boxes: Cadera, Rodilla,   │
│   Tobillo (izq/der)                  │
└──────────────────────────────────────┘
    │
    ▼  (crops generados con padding 15%)
    │
┌──────────────────────────────────────┐
│ Pose: YOLO-Pose (800 px)             │
│ telerx_pose_v2_ft/best.pt            │
│ → 8 keypoints anatómicos por crop    │
└──────────────────────────────────────┘
    │
    ▼  (proyección a coordenadas globales)
    │
┌──────────────────────────────────────┐
│ Cálculo CPAK                         │
│ aHKA = MPTA - LDFA                   │
│ JLO  = MPTA + LDFA                   │
│ → Clasificación en 9 fenotipos       │
└──────────────────────────────────────┘
```

### Scripts de inferencia

| Script | Propósito |
|--------|-----------|
| `13_final_inference.py` | Pipeline completo: detección → pose → CPAK |
| `13b_batch_inference.py` | Procesamiento por lotes (batch) |
| `30_streamlit_app.py` | Interfaz gráfica Streamlit |
| `35_screening_teleRx.py` | Solo detección (screening rápido) |

---

## Resumen de Métricas Finales

| Fase | Modelo | Metrica principal | Valor |
|------|--------|-------------------|-------|
| Detección | YOLO26s | mAP50 | **0.973** |
| Detección | YOLO26s | mAP50-95 | **0.677** |
| Pose | Pose v2 FT | Pose mAP50 | **0.983** |
| Pose | Pose v2 FT | Pose mAP50-95 | **0.981** |

---

*Documento generado el 2026-07-14. Los pesos de los modelos se encuentran en `training_runs/`.*
