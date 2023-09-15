# neural_desafio

## Punto 1: ¿Qué modelo elegir?
Me decidí por utilizar la regresión logística por dos razones fundamentales. En primer lugar, la variable que deseamos predecir es de naturaleza binaria, es decir, se trata de predecir si ocurre un "atraso" o "no atraso". La regresión logística es especialmente adecuada para abordar este tipo de situaciones.

En segundo lugar, y quizás aún más importante, la base de datos, una vez que se ha balanceado, no es de gran tamaño.

## Punto 2: Mejoramiento del modelo
Realicé pruebas utilizando diversas proporciones para la cantidad de datos de la clase sobre-representada, "no atraso", variando desde 0.5 hasta 1.5 veces la cantidad de datos de la clase sub-representada, "atraso". Después de estas pruebas, se determinó que la proporción que logró el mejor equilibrio entre la predicción de ambas clases fue de 1.0. Con esa proporción se logró una tasa de predicción de poco más del 60% para cada clase. El resultado por sí solo no es bueno, pero es una mejora respecto al modelo anterior en donde la clase sub-representada tenía tasas de predicción casi nulas.

Además, llevé a cabo experimentos adicionales al agregar o eliminar atributos en la base de datos. Incluso probé utilizar el modelo XGBoost. Sin embargo, ninguno de estos enfoques mostró un comportamiento significativamente mejor en términos de predicción.

Todos los experimentos fueron hechos usando MLFlow, que permitió comparar rápidamente los distintos modelos.

## Punto 3: Serializar el modelo mejorado
Utilicé la librería "pickle" para guardar el modelo predictor. Además, desarrollé un contenedor Docker con la consideración de futuros pasos, como desplegar el servicio en la nube, con el objetivo de evitar problemas de compatibilidad.

Se crearon tres instancias diferentes para manejar distintos tipos de llamadas a la API, cada una de las cuales se puede probar en el archivo "llamando_api.py":

*Llamada directa a la API*: En esta instancia, el input debe ser un DataFrame con un formato similar al utilizado durante el entrenamiento del modelo. Este DataFrame debe contener las características de cada vuelo que se desea predecir, con una fila por vuelo. La instancia devuelve la predicción de si el vuelo estará "atrasado" (1) o "no atrasado" (0).

*/retrain/*: Esta instancia permite reentrenar el modelo de predicción utilizando una nueva base de datos, que debe ser proporcionada como entrada. Una vez que el modelo se ha actualizado, será el nuevo modelo utilizado para realizar predicciones en la instancia anterior.

*/demo/*: Esta instancia se configura con datos de un vuelo precargados en el formato de DataFrame necesario para realizar pruebas de latencia en el servidor. Esto facilita la realización de pruebas y demostraciones del funcionamiento del sistema.

Todas estas instancias están diseñadas para automatizar el proceso de entrenamiento del modelo y la realización de predicciones, lo que facilita su implementación y uso en un entorno de producción.

## Punto 4: Automatizar y subir a servicios cloud
En cuanto a la implementación en un entorno de nube, mi enfoque se centró en subir una imagen Docker para crear un contenedor. Este contenedor necesitaba gestionar datos que no fueran efímeros, es decir, datos que no se perdieran cuando el contenedor se detuviera. Para abordar esta necesidad, utilicé "volúmenes" dentro del contenedor.

Sin embargo, me encontré con un desafío importante, ya que no todos los servicios de alojamiento en la nube admiten contenedores con volúmenes. Después de realizar diversas pruebas y evaluaciones, identifiqué la necesidad de encontrar un servicio que fuera estable, permitiera el uso de contenedores con volúmenes, proporcionara acceso al contenedor a través de la API y, preferiblemente, ofreciera una opción gratuita o una prueba gratuita.

Después de investigar varias opciones, decidí utilizar Vultr como el servicio de alojamiento en la nube final para mi contenedor. Vultr cumplió con todos los requisitos mencionados, lo que lo convirtió en la elección adecuada para alojar el contenedor y garantizar la estabilidad y la disponibilidad de los datos no efímeros necesarios para mi aplicación.

## Punto 5: Pruebas de estrés
Realicé pruebas de rendimiento utilizando la herramienta wrk, pero los resultados fueron insatisfactorios ya que no logré alcanzar los 50,000 solicitudes en un período de 45 segundos. Esto podría deberse a las limitaciones del servidor gratuito que estuve utilizando. Puedes consultar los detalles de los resultados en el archivo "wrk-test.jpg".

Para mejorar las métricas de rendimiento, es importante considerar varias estrategias de optimización del código, evaluación de recursos de servidor y exploración de alternativas de alojamiento en la nube.
