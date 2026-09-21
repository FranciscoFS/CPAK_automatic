Desarrollo y validación de un algoritmo de Deep Learning para la medición automatizada del eje coronal en telerradiografías de extremidades inferiores

Autores: Francisco Fernández, Raúl Zilleruelo, Joaquín Steinsapir, Catalina Vidal, Cristian Ruz y Pablo Besa.

**Introducción:** La evaluación del eje coronal es un paso fundamental en la planificación de cirugía de rodilla. Para ello se miden en telerradiografías (TelRx) de extremidades inferiores (EEII) distintos ángulos para fenotipar los perfiles coronales. Este proceso demanda tiempo, requiere entrenamiento y está sujeto a variabilidad interobservador.

**Objetivo:** Desarrollar y validar un algoritmo de deep learning para analizar TelRx de EEII y evaluar su concordancia clínica con evaluadores expertos.

**Método:** Se desarrolló un algoritmo basado en YOLO, estructurado en fases secuenciales: localización de cadera, rodilla y tobillo, seguida de la identificación de puntos anatómicos para calcular HKA, mLDFA y mMPTA. Se entrenó con 280 radiografías para detección y 1.734 recortes anatómicos para la localización de puntos. La concordancia se evaluó mediante el coeficiente de correlación intraclase (ICC) con IC95% a partir de 57 TelRx (112 mediciones) evaluadas por dos expertos, comparando la IA contra el promedio de ambos. Además, se calculó el error sistemático mediante Bland-Altman.

**Resultados:** El modelo mostró una excelente validación interna (mAP50 > 97% en detección y > 98% en puntos anatómicos) y una velocidad de procesamiento de 0,72 segundos por Rx. La concordancia fue excelente para el HKA (ICC=0,995; IC95%: 0,99–1,00) y buena para el mLDFA (ICC=0,873; IC95%: 0,83–0,91) y el mMPTA (ICC=0,873; IC95%: 0,80–0,92). El error sistemático fue inferior a 1° en todas las mediciones (HKA: +0,20°; mLDFA: −0,15°; mMPTA: −0,81°).

**Conclusión:** El modelo mostró un excelente rendimiento y velocidad de procesamiento, con una concordancia excelente en la alineación global y buena en las mediciones focales, con un error clínicamente no significativo. Estos hallazgos sugieren que los modelos de visión computacional pueden estandarizar y optimizar el análisis de TelRx.
