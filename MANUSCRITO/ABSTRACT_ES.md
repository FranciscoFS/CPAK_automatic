# Resumen

## Introducción
La evaluación del eje coronal es un paso fundamental en la planificación de la cirugía de rodilla. Para ello se miden manualmente, sobre teleradiografías de extremidades inferiores, el ángulo cadera-rodilla-tobillo (HKA), el ángulo lateral distal femoral mecánico (mLDFA) y el ángulo medial proximal tibial mecánico (mMPTA). Este proceso demanda tiempo, requiere entrenamiento y está sujeto a variabilidad entre observadores. El objetivo del estudio fue desarrollar un algoritmo basado en inteligencia artificial (IA) para automatizar estas mediciones y evaluar su concordancia con la medición manual.

## Método
Se desarrolló un algoritmo con YOLO en dos etapas. En la primera, un modelo de detección localizó cadera, rodilla y tobillo en 280 TeleRx, divididas en entrenamiento (n=197), validación (n=55) y prueba (n=28). En la segunda, un modelo para localizar reparos anatómicos identificó los puntos necesarios para calcular el HKA, el mLDFA y el mMPTA. Este último modelo se entrenó inicialmente con 943 recortes etiquetados y luego se afinó con un conjunto ampliado de 1.734 recortes, utilizando sus pesos previos como punto de partida. Para evaluar la concordancia del algoritmo, tres observadores midieron manualmente 57 TeleRx (104 Rodillas) no utilizadas durante el entrenamiento. Se calculó el coeficiente de correlación intraclase (ICC) con su intervalo de confianza del 95% (IC95%) entre la IA y cada observador, así como entre los propios observadores.

## Resultados
En los conjuntos de evaluación, el modelo de detección alcanzó un mAP50 de 0,973 y un mAP50-95 de 0,677. El modelo para localizar reparos anatómicos obtuvo un mAP50 de 0,983 y un mAP50-95 de 0,981. El procesamiento de las 57 TeleRx tomó 41,9 segundos, equivalente a 0,74 segundos por imagen. La concordancia entre la IA y el observador fue excelente para el HKA (ICC=0,984; IC95%: 0,98–0,99), buena para el mLDFA (ICC=0,856; IC95%: 0,80–0,90) y buena para el mMPTA (ICC=0,819; IC95%: 0,74–0,87). El sesgo fue inferior a 0,6° en las tres métricas.

## Discusión y conclusión
El desempeño de ambos modelos fue alto y el tiempo de procesamiento bajo. La concordancia entre la IA y la medición manual fue excelente para el HKA y muy buena para el mLDFA y el mMPTA, con un sesgo mínimo en todos los casos. Estos resultados sugieren que la herramienta podría constituir un apoyo objetivo y eficiente para la planificación quirúrgica de la rodilla.
