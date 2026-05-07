Especificación Técnica: Clasificación Automatizada CPAK con YOLO26 Pose



Este documento define la arquitectura de datos y la lógica clínica para automatizar el cálculo del fenotipo de alineamiento coronal de rodilla (CPAK) mediante Deep Learning local.



\# 1\\. Definición del Esqueleto Único (7 Puntos)



Se utiliza un esqueleto maestro de 7 puntos clave. En recortes específicos, los puntos no presentes deben marcarse con visibilidad 0.



| ID  | Landmark Anatómico        | Crop Destino | Función en CPAK                     |

| --- | ------------------------- | ------------ | ----------------------------------- |

| 0   | Centro Cabeza Femoral     | Cadera       | Origen del eje mecánico femoral.    |

| 1   | Centro Espinas Tibiales   | Rodilla      | Punto de control central (Bisagra). |

| 2   | Centro Domo del Talo      | Tobillo      | Fin del eje mecánico tibial.        |

| 3   | Cóndilo Medial (Vértice)  | Rodilla      | Tangente femoral distal (LDFA).     |

| 4   | Cóndilo Lateral (Vértice) | Rodilla      | Tangente femoral distal (LDFA).     |

| 5   | Plateau Medial (Centro)   | Rodilla      | Tangente tibial proximal (MPTA).    |

| 6   | Plateau Lateral (Centro)  | Rodilla      | Tangente tibial proximal (MPTA).    |



\# 2\\. Configuración del Dataset (data.yaml)



path: ./dataset\_cpak\_local  

train: train/images  

val: valid/images  

<br/>kpt\_shape: \\\[7, 3\\] # \\\[x, y, visibility\\]  

names:  

0: Hip\_Crop  

1: Knee\_Crop  

2: Ankle\_Crop



\# 3\\. Lógica de Etiquetado por Imagen



\- Para el entrenamiento sobre recortes (crops), se debe seguir la siguiente visibilidad (v):

\- Crop de Cadera: Punto 0 (v=2). Puntos 1-6 (v=0).

\- Crop de Rodilla: Puntos 1, 3, 4, 5, 6 (v=2). Puntos 0 y 2 (v=0).

\- Crop de Tobillo: Punto 2 (v=2). Puntos 0, 1 y 3-6 (v=0).



\# 4\\. Algoritmo de Cálculo (Trigonometría)



Una vez obtenidas las coordenadas globales (proyectadas desde el crop a la Rx original):



\- Eje Mecánico Femoral: Vector entre Centro de Cadera (P0) y Centro de Rodilla (P1).

\- Eje Mecánico Tibial: Vector entre Centro de Rodilla (P1) y Centro de Tobillo (P2).

\- LDFA: Ángulo lateral entre el eje femoral y la tangente femoral (P3-P4).

\- MPTA: Ángulo medial entre el eje tibial y la tangente tibial (P5-P6).



\# 5\\. Matriz de Clasificación CPAK



Métricas clave:



\- aHKA = MPTA - LDFA

\- JLO = MPTA + LDFA



| Métrica | Varo / Apex Distal | Neutro      | Valgo / Apex Proximal |

| ------- | ------------------ | ----------- | --------------------- |

| aHKA    | < -2°              | \\-2° a 2°   | \\> 2°                 |

| JLO     | < 177°             | 177° a 181° | \\> 181°               |



\# 6\\. Recomendaciones de Entrenamiento Local (YOLO26)



\- Optimizador: Utilizar MuSGD para mayor estabilidad en keypoints.

\- Resolución (imgsz): Entrenar a 1024 para asegurar precisión sub-píxel en los bordes articulares.

\- Augmentation: Mantener fliplr: 0.5 para que el modelo aprenda indistintamente de la lateralidad de la extremidad.

