# TeleRx: Clasificación Automatizada CPAK con YOLO

Este proyecto implementa un sistema avanzado de visión computacional médico basado en modelos YOLO (YOLOv8 y YOLO-Pose) para el análisis automatizado de radiografías de extremidades inferiores (Telemetrías o TeleRx).

El objetivo final del proyecto es calcular de forma automática el fenotipo de alineamiento coronal de rodilla (**CPAK**) detectando puntos clave (Keypoints) en la cadera, rodilla y tobillo mediante Deep Learning.

## Fases del Proyecto

1. **Fase 1 (Object Detection):** Mediante YOLOv8, localiza las regiones de interés (Bounding Boxes) para Cadera, Rodilla y Tobillo, generando recortes (crops) para su posterior análisis. Incluye algoritmos de auto-tagueo y simetría.
2. **Fase 2 (Pose / Keypoint Detection):** Mediante un modelo de pose, ubica un esqueleto maestro de 7 puntos anatómicos sobre los crops obtenidos en la Fase 1. A partir de esos puntos, se aplica trigonometría para calcular el Eje Mecánico Femoral/Tibial, el LDFA, el MPTA y, finalmente, las métricas CPAK (aHKA y JLO).

## Instalación y Entorno de Desarrollo

Es altamente recomendable utilizar un entorno virtual (Conda o venv) para evitar conflictos de dependencias. 

### Opción 1: Usando Conda (Recomendado)

Conda facilita la instalación de las versiones correctas de PyTorch para tu GPU.

```bash
# 1. Crear el entorno conda
conda create -n telerx-yolo python=3.10 -y

# 2. Activar el entorno
conda activate telerx-yolo

# 3. Instalar PyTorch con soporte CUDA (Ajusta la versión según tu entorno)
# (Ejemplo para CUDA 11.8)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 4. Instalar el resto de dependencias
pip install -r requirements.txt
```

### Opción 2: Usando `venv` (Python estándar)

```bash
# 1. Crear el entorno virtual
python -m venv venv

# 2. Activar el entorno
# En Windows (PowerShell):
.\venv\Scripts\activate
# En Linux/Mac:
source venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt
```

## Estructura de Scripts (`scripts/`)

El flujo de trabajo se controla mediante varios scripts:
- `1_split_data.py`: Divide el dataset manual o validado en Train/Val/Test (70/20/10).
- `2_train_yolo.py`: Ejecuta el entrenamiento del modelo de object detection.
- `3_auto_tag.py`: Motor de etiquetado automático en nuevas imágenes (con lógica de simetría y exclusión de bajas confianzas).
- `4_audit_labels.py`: Analiza las etiquetas generadas y arroja reportes estadísticos.
- `5_visualize_results.py`: Crea mosaicos con Bounding Boxes para revisión humana visual.
- `6_generate_crops.py`: Extrae y normaliza las regiones articulares para la Fase 2 (Keypoints).

## Notas Adicionales
- Asegúrate de colocar las imágenes crudas en el directorio `data/` antes de ejecutar los pipelines de auto-tagueo.
- Las particiones de dataset entrenables se generan en `dataset/`.
- Los pesos resultantes de entrenamientos se guardan en `.pt` dentro de `training_runs/`.

> **Colaboración**: Evita subir datasets pesados o modelos (.pt) al repositorio, ya que están incluidos en el `.gitignore`.
