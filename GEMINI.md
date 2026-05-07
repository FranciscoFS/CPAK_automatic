# TeleRx YOLO Project: Identificación de Estructuras Óseas

Este proyecto utiliza **YOLOv8** (Ultralytics) para la detección y etiquetado automático de articulaciones (Cadera, Rodilla, Tobillo) en radiografías TeleRx.

## Estructura del Proyecto

- `data/`: Imágenes raw (sin taguear o pendientes de revisión).
- `CPAK.yolo26/`: Dataset original exportado de Roboflow.
- `dataset/`: Dataset organizado (70/20/10) listo para entrenamiento.
- `scripts/`: Herramientas de automatización.
- `auto_tagged_labels/`: Etiquetas generadas automáticamente (.txt y .meta).
- `visualizations/`: Ejemplos visuales y mosaicos de los resultados.
- `training_runs/`: Logs y pesos (.pt) del entrenamiento.

## Scripts y Workflows

### 1. Preparación de Datos (`scripts/1_split_data.py`)
Divide las imágenes tagueadas manualmente en carpetas de entrenamiento, validación y prueba bajo una proporción 70/20/10.

### 2. Entrenamiento (`scripts/2_train_yolo.py`)
Entrena un modelo **YOLOv8n** utilizando la GPU (RTX 3050).
- **Parámetros:** 100 épocas, 640px de resolución.
- **Salida:** `training_runs/telerx_yolov8n/weights/best.pt`.

### 3. Auto-Tagueo Inteligente (`scripts/3_auto_tag.py`)
Genera etiquetas para imágenes nuevas con las siguientes características:
- **Filtro Top-1:** Solo una detección (la de mayor confianza) por clase.
- **Lógica de Simetría:** Si falta un lado (ej. Cadera Izq) y el otro tiene alta confianza, se genera el par por reflejo horizontal.
- **Metadatos:** Crea archivos `.meta` para rastrear si la etiqueta fue detección (D) o simetría (S).

### 4. Auditoría y Reportes (`scripts/4_audit_labels.py`)
Analiza la calidad del tagueo automático.
- Genera un resumen estadístico en consola.
- Crea `consolidated_audit.csv` para análisis detallado en Excel.

### 5. Visualización de Resultados (`scripts/5_visualize_results.py`)
Genera muestras visuales con Bounding Boxes:
- **Escalado Dinámico:** Ajusta grosor y texto según la resolución de la imagen.
- **Etiquetas de Confianza:** Muestra el % o la marca de "SIMETRÍA".
- **Mosaico 5x5:** Genera una cuadrícula de 25 imágenes para revisión rápida.

### 6. Normalización de Nombres (`scripts/5b_normalize_roboflow_names.py`)
Limpia los nombres de archivos descargados de Roboflow (elimina el sufijo `_jpg.rf.[hash]`) para que coincidan exactamente con la raw data original.

### 7. Generación de Recortes (`scripts/6_generate_crops.py`)
Crea imágenes individuales para cada articulación detectada:
- **Padding:** Añade un 15% de margen extra para facilitar el tagueo de Keypoints.
- **Nomenclatura:** Formato `Paciente_Clase.jpg`.
- **Uso:** Estos recortes se suben a un nuevo proyecto de Roboflow para la Fase 2.

## Fase 2: Keypoint Detection (Esqueletos)
El objetivo es pasar de simples cajas a puntos clave exactos en los huesos.
1. Ejecutar la normalización si se descarga data nueva de Roboflow.
2. Generar los recortes con el Script 6.
3. Subir los recortes a un proyecto de tipo "Keypoint Detection" en Roboflow.

## Instrucciones para el Desarrollador

1. Siempre activar el ambiente conda: `conda activate fxedr-yolo`.
2. Para mejorar el modelo, añadir imágenes con errores detectados en el audit al set manual y re-entrenar.
3. El umbral de confianza actual para auto-tagueo es **0.20** para maximizar la captura de articulaciones difíciles.

---
*Nota: Este archivo define la arquitectura y convenciones del repositorio.*
