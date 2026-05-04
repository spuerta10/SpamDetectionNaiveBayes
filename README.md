<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/e4/Logos%C3%ADmbolo_UPB.svg/960px-Logos%C3%ADmbolo_UPB.svg.png" alt="Naive Bayes theorem" width="300" />

| | |
|-|-|
| Santiago Puerta | santiago.puertaf@upb.edu.co |
| Iker Acevedo Vargas | iker.acevedo@upb.edu.co |

# Indice
- 1. [Resumen](#resumen)
- 2. [Introducción](#intro)
- 3. [Contexto](#contexto)
    - 3.1 [Bernoulli](#bernoulli)
    - 3.2 [Multinomial](#multinomial)
        - 3.2.1 [Ejemplo práctico](#ejemplo-practico)
- 4. [Estado del arte](#estado-del-arte)
    - 4.1 [Metsis et al. (2006)](#metsis-et-al-2006)
    - 4.2 [Mohammed et al. (2013)](#mohammed-et-al-2013)
    - 4.3 [Hammad & El-Halees (2013)](#hammad-el-halees-2013)
    - 4.4 [Fusilier et al. (2015)](#fusilier-et-al-2015)
    - 4.5 [Wu et al. (2017)](#wu-et-al-2017)
    - 4.6 [Aiyar & Shetty (2018)](#aiyar-shetty-2018)
    - 4.7 [Mani et al. (2018)](#mani-et-al-2018)
    - 4.8 [Saeed et al. (2019)](#saeed-et-al-2019)
    - 4.9 [Kontsewaya et al. (2021)](#kontsewaya-et-al-2021)
    - 4.10 [Wang (2024)](#wang-2024)
    - 4.11 [Discusión](#discusion)
- 5. [Resultados](#resultados)
- 6. [Consideraciones éticas](#consideraciones-eticas)
- 7. [Conclusiones](#conclusiones)
- 8. [Bibliografia](#bibliografia)

<a id="resumen"></a>
# 1. Resumen

EL spam representa una amenaza constante para las comunicaciones por correo electrónico, millones de mensajes fraudulentos son enviados diariamente. Este trabajo compara el desempeño de tres variantes de Naive Bayes (Multinomial, Multinomial Binario y complement) frente a un Perceptrón Multicapa (MLP) en la detección de spam sobre corpus Enron, compuesto por 33.716 correos electrónicos. Los modelos fueron evaluados mediante validación cruzada con énfasis en recall, dado el costo asimétrico de los falsos negativos en esta tarea. Los resultados muestran que el MultinomialNB alcanza el mayor recall (0.9985), mientras que MLP obtiene el mejor balance con un F1-score de (0.9930). Se concluye que la complejidad del modelo no garantiza mejor desempeño y que Naive Bayes representa una alternativa eficiente y competitiva para esta tarea.

**Palabras clave:** detección de spam, Naive Bayes, Multinomial NB, Complement NB, Perceptrón Multicapa, clasificación de texto, TF-IDF, procesamiento de lenguaje natural, corpus Enron, aprendizaje automático.


<a id="intro"></a>
# 2. Introducción

El crecimiento exponencial de la comunicación digital ha traído consigo un aumento proporcional en la cantidad de mensajes no deseados (spam); se estima que “cada día se envían 160 mil millones de correos electrónicos no deseados” [1]. Asimismo, un agravante de esta situación se presenta tanto en instituciones financieras como en servicios de entrega. Las primeras, al ser objetivos frecuentes de los atacantes, llegan a concentrar “el 27% de los mensajes fraudulentos” [1]. En el caso de los servicios de entrega, el problema es diferente: alrededor de 1.100 millones de mensajes fraudulentos estuvieron relacionados con este tema [1], lo que genera una pérdida de confianza por parte de los usuarios y perjudica la reputación de estas empresas de cara a potenciales usuarios futuros.

Diversos enfoques han sido propuestos para abordar este problema, abarcando desde reglas heurísticas [2] y modelos clásicos de aprendizaje automático como Naive Bayes y Support Vector Machines [2], hasta arquitecturas más complejas basadas en redes neuronales, como Multi Layer Perceptron y modelos de lenguaje como BERT [3]. Sin embargo, a pesar de esta diversidad de enfoques y niveles de complejidad, surge una pregunta fundamental: ¿implica un mayor nivel de complejidad en el modelo un mejor rendimiento?

El presente artículo busca dar respuesta a esta pregunta mediante la comparación del desempeño de una familia de modelos simples, como lo son los Naive Bayes, frente a un modelo más complejo como Multi Layer Perceptron, aplicados a la tarea de detección de spam. Se analizará no solo la calidad de las predicciones, sino también las implicaciones en términos de eficiencia y aplicabilidad en escenarios reales.

<a id="contexto"></a>
# 3. Contexto

Cuando hablamos de Naive Bayes, no nos referimos a un algoritmo en particular, sino, más bien a una familia de algoritmos con diversas variantes: Bernoulli, Complement, Multinomial, entre otros, los cuales han sido ampliamente investigados y usados en la industria para la detección de spam.

Su alta adopción en este tipo de aplicaciones se debe a múltiples factores. En primer lugar, estos modelos se caracterizan por su simplicidad y facilidad de implementación, lo que permite su integración en sistemas productivos. Asimismo, presentan una baja complejidad computacional tanto en entrenamiento como en predicción, resultando ideales para el procesamiento de grandes volúmenes de datos textuales [2]-[4], por lo que suelen ser preferidos frente a modelos más complejos como Support Vector Machines o métodos de boosting [2]-[4].

¿A qué se debe su alto rendimiento en esta tarea? Los modelos de Naive Bayes, en sus diferentes variantes, se basan en el Teorema de Bayes y asumen independencia condicional entre las variables, lo que les permite calcular probabilidades para cada característica de manera eficiente. En la detección de spam, esto se traduce en estimar la probabilidad de que un mensaje sea spam a partir de las palabras encontradas en el mensaje de la siguiente forma:

$$
Posterior = \frac{Likelihood\cdot Prior}{Evidence}
$$

Donde el Likelihood representa la probabilidad de observar la combinación específica de palabras que componen el mensaje dentro de una clase determinada ($Spam$ o $Ham$). El Prior constituye la probabilidad base de cada clase en el dataset. Por su parte, el Evidence es la probabilidad total de encontrar ese conjunto de términos coexistiendo en un mismo mensaje, calculada de forma independiente a la clase; es decir, es la suma de las probabilidades de observar dicha combinación a través de todas las categorías posibles. Finalmente, el Posterior es la probabilidad de que, dada esa agrupación particular de palabras, el mensaje pertenezca a una clase específica.

Expresada en una notación matemática, obtenemos la siguiente fórmula:

$$
P(C \mid X) = \frac{P(X \mid C)\,P(C)}{P(X)}
$$

Donde $P(x)$ o **evidence** se calcula de la siguiente forma: 

$$
P(x)=P(x\mid \text{Spam})\,P(\text{Spam})+P(x\mid \text{No Spam})\,P(\text{No Spam})
$$

Si bien estas fórmulas definen el marco general de los modelos bayesianos, su aplicación depende de cómo se modelan las características del mensaje. Con el objetivo de facilitar la comprensión de la base conceptual de estos algoritmos, iniciaremos analizando la variante Bernoulli, la cual sobresale en la detección de spam basada en la presencia o ausencia de palabras clave.

<a id="bernoulli"></a>
## 3.1 Bernoulli

Supongamos que tenemos un conjunto de 100 correos electrónicos, clasificados de la siguiente manera:
- 80 son Ham $P(\text{Ham}) = 0.8$
- 20 son Spam $P(\text{Spam}) = 0.2$ 

Esta distribución inicial representa nuestro Prior.

Imaginemos ahora que recibimos un nuevo mensaje con la combinación de palabras: "Premio Ganaste". 

El mensaje se representa como un vector binario donde cada palabra toma el valor de 1 si está presente y 0 en caso contrario. Por ejemplo, para un vocabulario como:

$$
\begin{bmatrix}
\text{premio} \\
\text{ganaste} \\
\text{gratis} \\
\text{dinero} \\
\text{oferta}
\end{bmatrix}
$$

el mensaje “Premio Ganaste” se representaría como:

$$
(1,1,0,0,0)
$$

Para clasificarlo, el modelo sigue este proceso:

Cálculo del Likelihood (Verosimilitud): El modelo evalúa la probabilidad de observar estas palabras juntas dentro de la categoría Spam. Al asumir independencia, multiplicamos la probabilidad de encontrar la palabra premio por la probabilidad de encontrar la palabra ganaste, ambas condicionadas a que el mensaje sea Spam:

$$
P(\text{premio} \mid \text{Spam}) \cdot P(\text{ganaste} \mid \text{Spam}) \cdot (1 - P(\text{gratis} \mid \text{Spam})) \cdot (1 - P(\text{dinero} \mid \text{Spam})) \cdot (1 - P(\text{oferta} \mid \text{Spam}))
$$

Cálculo del Numerador: Posteriormente, este resultado se multiplica por el *Prior* de la clase para ponderar qué tan común es el Spam en general:

$$
[P(\text{premio} \mid \text{Spam}) \cdot P(\text{ganaste} \mid \text{Spam}) \cdot (1 - P(\text{gratis} \mid \text{Spam})) \cdot (1 - P(\text{dinero} \mid \text{Spam})) \cdot (1 - P(\text{oferta} \mid \text{Spam}))] \cdot P(\text{Spam})
$$

Normalización mediante el Evidence: Finalmente, dividimos por el *Evidence*. Este componente representa la probabilidad total de observar la combinación premio y ganaste en todo nuestro universo de correos, sumando su probabilidad de aparición tanto en la clase Spam como en la clase Ham (No Spam):

$$
P(\text{Spam} \mid \text{mensaje}) = \frac{[P(\text{premio} \mid \text{Spam}) \cdot P(\text{ganaste} \mid \text{Spam}) \cdot (1 - P(\text{gratis} \mid \text{Spam})) \cdot (1 - P(\text{dinero} \mid \text{Spam})) \cdot (1 - P(\text{oferta} \mid \text{Spam}))] \cdot P(\text{Spam})}{P(\text{X})}
$$

Donde el *Evidence* se desglosa como la suma de las probabilidades conjuntas de ambas clases:

$$
P(X) =
[P(\text{premio} \mid \text{Spam}) \cdot P(\text{ganaste} \mid \text{Spam}) \cdot (1 - P(\text{gratis} \mid \text{Spam})) \cdot (1 - P(\text{dinero} \mid \text{Spam})) \cdot (1 - P(\text{oferta} \mid \text{Spam})) \cdot P(\text{Spam})]
+
[P(\text{premio} \mid \text{Ham}) \cdot P(\text{ganaste} \mid \text{Ham}) \cdot (1 - P(\text{gratis} \mid \text{Ham})) \cdot (1 - P(\text{dinero} \mid \text{Ham})) \cdot (1 - P(\text{oferta} \mid \text{Ham})) \cdot P(\text{Ham})]
$$

<a id="multinomial"></a>
## 3.2 Multinomial

Sin embargo, Bernoulli presenta algunas limitaciones:

- Representa cada mensaje como un vector binario, indicando si una palabra del mensaje está presente o ausente en el vocabulario. Esto implica que, independientemente de si el mensaje es corto o largo, el modelo debe evaluar la presencia o ausencia de cada término del vocabulario [4]. Por ejemplo, si el vocabulario está compuesto por 1000 palabras, para el mensaje "Premio Ganaste" se deberá construir y evaluar un vector de 1000 posiciones.

- Supone independencia condicional entre todas las palabras que componen el mensaje, ignorando la frecuencia con la que aparece un término. Esta suposición resulta simplista, ya que la coocurrencia de palabras en un mismo mensaje no es un fenómeno aislado en el caso del spam [4].

- Tiene en cuenta los términos del vocabulario que no aparecen en el mensaje al calcular la probabilidad, lo cual puede llevar a que el producto de las probabilidades asociadas a términos ausentes sesgue el resultado [4].

Para mitigar dichas limitaciones, surge la variante **Multinomial**.

A diferencia de Bernoulli, este enfoque no modela únicamente la presencia o ausencia de las palabras, sino que tiene en cuenta la frecuencia con la que cada término aparece en el mensaje. De esta forma, un mensaje donde una palabra relevante aparece múltiples veces tendrá un mayor impacto en la probabilidad final.

En lugar de representar los mensajes como vectores binarios, Multinomial utiliza representaciones basadas en conteos o ponderaciones, como Bag of Words o TF-IDF. Esto permite trabajar de manera más eficiente con textos de distinta longitud, evitando la necesidad de evaluar explícitamente la ausencia de todos los términos del vocabulario.

Desde el punto de vista probabilístico, el likelihood se calcula como el producto de las probabilidades de cada término elevado al número de veces que aparece en el mensaje, lo que permite incorporar directamente la información de frecuencia:

$$
P(X \mid C) = \prod_{i=1}^{n} P(x_i \mid C)^{f_i}
$$

donde $f_i$ representa la frecuencia del término $x_i$ en el mensaje.

Adicionalmente, al centrarse en los términos presentes en el mensaje, este enfoque reduce el impacto que tienen las palabras ausentes en el cálculo de la probabilidad.

<a id="ejemplo-practico"></a>
### 3.2.1 Ejemplo práctico

Supongamos el mismo escenario inicial:

- $P(\text{Ham}) = 0.8$
- $P(\text{Spam}) = 0.2$

Imaginemos ahora un nuevo mensaje:

"Premio Premio Ganaste"

A diferencia de Bernoulli, en este caso cada palabra se representa según su frecuencia dentro del mensaje. Por ejemplo, considerando el vocabulario 

$$
\begin{bmatrix}
\text{premio} \\
\text{ganaste} \\
\text{gratis} \\
\text{dinero} \\
\text{oferta}
\end{bmatrix}
$$

el mensaje se representaría como:

$$
(2, 1, 0, 0, 0)
$$

Para clasificarlo, el modelo sigue el siguiente proceso:

Cálculo del Likelihood (Verosimilitud): El modelo evalúa la probabilidad de observar este mensaje dentro de la clase Spam. A diferencia de Bernoulli, cada probabilidad se eleva a la frecuencia del término:

$$
P(\text{premio} \mid \text{Spam})^2 \cdot P(\text{ganaste} \mid \text{Spam})^1
$$

Cálculo del Numerador: Se multiplica el resultado anterior por el *Prior* de la clase:

$$
[P(\text{premio} \mid \text{Spam})^2 \cdot P(\text{ganaste} \mid \text{Spam})] \cdot P(\text{Spam})
$$

Normalización mediante el Evidence: Finalmente, se divide por el *Evidence*, el cual se calcula como la suma de las probabilidades del mensaje en cada clase:

$$
P(\text{Spam} \mid \text{mensaje}) = 
\frac{
P(\text{premio} \mid \text{Spam})^2 \cdot P(\text{ganaste} \mid \text{Spam}) \cdot P(\text{Spam})
}{
P(X)
}
$$

Donde el *Evidence* se define como:

$$
P(X) =
[P(\text{premio} \mid \text{Spam})^2 \cdot P(\text{ganaste} \mid \text{Spam}) \cdot P(\text{Spam})]
+
[P(\text{premio} \mid \text{Ham})^2 \cdot P(\text{ganaste} \mid \text{Ham}) \cdot P(\text{Ham})]
$$

Nótese que, a diferencia de Bernoulli, la repetición del término "premio" incrementa su influencia en la probabilidad final, reflejando la importancia de la frecuencia en este modelo.

Algunas limitaciones presentadas por este modelo son las siguientes:
- Sigue presentando la suposición de independencia condicional entre palabras, explicada anteriormente.
- La relevancia de una palabra que aparece múltiples veces en un mensaje varía dependiendo de la longitud del mismo [4], ya que, a medida que el texto es más largo, su influencia relativa se diluye entre el resto de términos.
- Su rendimiento no mejora de forma proporcional con la cantidad de datos de entrenamiento, debido a las fluctuaciones en la proporción de mensajes y a los cambios en los temas de spam [4]. Esto implica una inhabilidad por parte del modelo para captar de manera eficiente cambios drásticos en la redacción de los mensajes de spam.

<a id="estado-del-arte"></a>
# 4. Estado del arte

La detección de spam ha evolucionado significativamente en las últimas décadas, pasando de enfoques basados en reglas heurísticas a modelos de aprendizaje automático y, más recientemente, a arquitecturas de aprendizaje profundo. En este contexto, los modelos basados en Naive Bayes han sido ampliamente estudiados en la literatura debido a su simplicidad, eficiencia y desempeño competitivo en tareas de clasificación de texto [2].

Con el fin de sintetizar estos aportes, a continuación se presentan algunos trabajos representativos en la literatura, donde se analizan los métodos utilizados y los resultados obtenidos en tareas de detección de spam.

<a id="metsis-et-al-2006"></a>
## 4.1 Metsis et al. (2006)

Analizan seis conjuntos de datos denominados Enron-Spam, con aproximadamente entre 5,000 y 6,000 mensajes cada uno, manteniendo el orden temporal y variando la proporción de spam para simular escenarios reales. Evalúan cinco variantes de Naive Bayes: Bernoulli multivariante, Multinomial con frecuencia de términos, Multinomial con atributos booleanos, Gaussiano y Flexible Bayes. Los resultados muestran que el Multinomial con atributos booleanos y el Flexible Bayes obtienen el mejor desempeño. No obstante, resaltan que Flexible Bayes presenta una mayor complejidad computacional, lo que limita su uso en sistemas de alta demanda. Concluyen que el Multinomial con atributos booleanos representa un balance adecuado entre precisión y eficiencia [4].

<a id="mohammed-et-al-2013"></a>
## 4.2 Mohammed et al. (2013)

Utilizan el dataset Email-1431 y seleccionan un subconjunto balanceado de 544 correos electrónicos para el entrenamiento. Proponen un enfoque basado en la generación de un léxico dinámico compuesto por palabras representativas de spam y ham. Evalúan modelos como Naive Bayes, Support Vector Machines, K-Nearest Neighbor, Árboles de Decisión y métodos basados en reglas. Encuentran que Naive Bayes y SVM son los clasificadores más efectivos. Como limitación, destacan que el uso de léxicos estáticos requiere actualizaciones constantes debido a la evolución del lenguaje utilizado en los mensajes de spam [5].

<a id="hammad-el-halees-2013"></a>
## 4.3 Hammad & El-Halees (2013)

Analizan reseñas de hoteles en árabe utilizando características de contenido, metadatos y atributos del autor. Evalúan modelos como K-Nearest Neighbor, Naive Bayes y Support Vector Machines, encontrando que Naive Bayes alcanza la mayor precisión, con un valor de 99.2%. Como limitación, señalan que el proceso de etiquetado de los datos se realizó de forma manual, lo que puede introducir sesgos en los resultados [6].

<a id="fusilier-et-al-2015"></a>
## 4.4 Fusilier et al. (2015)

Utilizan un conjunto de 1.600 reseñas de hoteles y proponen el uso de n-gramas de caracteres para la detección de spam. Evalúan modelos como SVM, Naive Bayes y Multinomial Naive Bayes, encontrando que este último obtiene el mejor desempeño en sus experimentos. Además, reportan que el uso de n-gramas de caracteres mejora los resultados frente a n-gramas de palabras entre un 2.1% y 2.3%. Como trabajo futuro, proponen combinar ambas representaciones para mejorar la calidad de los modelos [7].

<a id="wu-et-al-2017"></a>
## 4.5 Wu et al. (2017)

Emplean un dataset de Twitter que contiene más de 2 millones de mensajes, con una proporción significativa de spam. Proponen un enfoque basado en Deep Learning utilizando Word2Vec y Doc2Vec para aprender representaciones vectoriales de los textos. Evalúan modelos tradicionales como Naive Bayes y Complement Naive Bayes, así como modelos más complejos como Random Forest, Decision Tree y Multi Layer Perceptron. Encuentran que el enfoque basado en redes neuronales alcanza una precisión cercana al 95%, superando a los métodos tradicionales, cuyos resultados se sitúan entre el 80% y 85%. Asimismo, resaltan que los modelos bayesianos presentan una caída significativa en escenarios de datos desbalanceados [8].

<a id="aiyar-shetty-2018"></a>
## 4.6 Aiyar & Shetty (2018)

Analizan un conjunto de 13.000 comentarios de YouTube y proponen el uso de n-gramas de caracteres en lugar de palabras para mejorar la detección de spam. Evalúan modelos como Multinomial Naive Bayes, Random Forest y Support Vector Machines (SVM), encontrando que este último obtiene el mejor desempeño al utilizar 6-gramas de caracteres. Asimismo, evidencian que Naive Bayes presenta consistentemente el rendimiento más bajo entre los modelos evaluados, especialmente al incrementar el valor de n en los n-gramas. Como limitación, sugieren que el uso de representaciones más avanzadas, como embeddings de palabras, podría mejorar los resultados [9].

<a id="mani-et-al-2018"></a>
## 4.7 Mani et al. (2018)

Proponen un enfoque de aprendizaje por ensamble que combina Naive Bayes, Random Forest y Support Vector Machines para la detección de spam en reseñas en inglés. Utilizan n-gramas como base para la extracción de características y encuentran que esta combinación mejora la robustez del modelo, alcanzando una precisión de hasta el 87.68%. Destacan que los métodos de ensamble permiten compensar las limitaciones individuales de modelos como Naive Bayes [10].

<a id="saeed-et-al-2019"></a>
## 4.8 Saeed et al. (2019)

Utilizan dos conjuntos de datos de reseñas de hoteles en árabe y proponen un enfoque de aprendizaje por ensamble que combina clasificadores basados en reglas con modelos de aprendizaje automático. Su metodología incluye el uso de n-gramas y un módulo específico para el manejo de negaciones. Evalúan múltiples modelos, incluyendo Naive Bayes, SVM, KNN, Random Forest y redes neuronales. Encuentran que el enfoque de stacking alcanza los mejores resultados, con precisiones superiores al 95%. Asimismo, reportan que Naive Bayes obtiene un desempeño destacado cuando se integra dentro del ensamble [11].

<a id="kontsewaya-et-al-2021"></a>
## 4.9 Kontsewaya et al. (2021)

Utilizan un conjunto de 5,728 correos electrónicos en inglés obtenidos de Kaggle y proponen un enfoque basado en técnicas de procesamiento de lenguaje natural, incluyendo preprocesamiento, tokenización y extracción de características mediante CountVectorizer. Evalúan múltiples algoritmos, entre ellos Naive Bayes, K-Nearest Neighbors, Support Vector Machines, Regresión Logística, Árboles de Decisión y Random Forest. Los resultados muestran que la Regresión Logística y Naive Bayes alcanzan los niveles más altos de efectividad, con una precisión cercana al 99%. Como limitación, señalan que el rendimiento de Naive Bayes puede verse afectado cuando el mensaje contiene palabras no observadas durante el entrenamiento [12].

<a id="wang-2024"></a>
## 4.10 Wang (2024)

Realiza una revisión de sistemas modernos de filtrado de spam, comparando modelos de aprendizaje automático y aprendizaje profundo. Analiza algoritmos como Naive Bayes, Support Vector Machines, Random Forest, BERT y redes convolucionales. Encuentra que, aunque Naive Bayes destaca por su eficiencia y facilidad de implementación, es superado por modelos más complejos en tareas que requieren comprensión contextual. Además, identifica como limitaciones relevantes el concept drift y la aparición de spam generado mediante modelos de lenguaje [2].

<a id="discusion"></a>
## 4.11 Discusión

A partir de los trabajos revisados, se observa que los modelos basados en Naive Bayes continúan siendo ampliamente utilizados en la detección de spam, debido a su eficiencia y desempeño competitivo en distintos escenarios. Sin embargo, también se evidencian limitaciones relacionadas con la suposición de independencia condicional, la sensibilidad a cambios en la distribución de los datos y la incapacidad para capturar relaciones semánticas complejas. En este contexto, modelos más avanzados, como Support Vector Machines o arquitecturas de Deep Learning, tienden a obtener mejores resultados en escenarios más complejos, aunque a costa de un mayor costo computacional. Esto plantea la necesidad de analizar en qué medida el incremento en la complejidad del modelo se traduce en mejoras significativas en el desempeño, motivando así la comparación propuesta en este trabajo.

<a id="resultados"></a>
# 5. Resultados

## 5.1 Configuración del método

El preprocesamiento aplicado a cada correo incluyó: eliminación de cabeceras (Subject, From, To), URLs, direcciones de correo, caracteres especiales y números; seguido de tokenización, remoción de stop words y lematización mediante WordNetLemmatizer de NLTK. Las representaciones textuales se generaron con TF - IDF (unigramas y bigramas, máximo de 50.000 características) para MultinomialNB y ComplementNB, y con CountVectorizer binario para Multinomial Binary NB. El conjunto de entrenamiento corresponde al 80% del corpus Enron (33.716 correos), reservando el 20% para hacer la evaluación.

La búsqueda de hiperparámetros se realizó con GridSearchCV (5-fold, métrica: recall) para los modelos de Naive Bayes, y con RandomizedSearchCV para el MLP. Los mejores parámetros encontrados fueron:

| Modelo | Hiperparámetros óptimos |
|--------|------------------------|
| MultinomialNB | alpha=1.0, class_prior=[0.2, 0.8] |
| Multinomial Binary NB | alpha=0.001, class_prior=[0.2, 0.8] |
| ComplementNB | alpha=0.01, norm=False, class_prior=None |
| MLP | hidden_layers=(128,), lr=0.001, alpha=0.0001, activation=tanh |

## 5.2 Calidad obtenida

Los modelos fueron evaluados en el conjunto de pruebas con las métricas de accuracy, precision, recall y F1-score:

| Modelo | Accuracy | Precision | F1 | Recall |
|--------|----------|-----------|----|--------|
| MultinomialNB | 0.9828 | 0.9686 | 0.9834 | **0.9985** |
| Multinomial Binary NB | 0.9880 | 0.9881 | 0.9882 | 0.9884 |
| ComplementNB | 0.9901 | 0.9887 | 0.9903 | 0.9918 |
| MLP | **0.9929** | **0.9919** | **0.9930** | 0.9942 |

El MLP obtuvo el mejor balance general entre métricas. No obstante, MultinomialNB alcanzó el recall más alto (0.9985), lo que indica una mayor capacidad para detectar spam sin perder correos fraudulentos, a costa de una ligera reducción en precisión. El análisis de escalabilidad mostró que los modelos Naive Bayes presentan tiempos de entrenamiento significativamente menores que el MLP, lo que los hace más adecuados para entornos con grandes volúmenes de datos o recursos computacionales limitados.

## 5.3 Ejemplo de despliegue

El modelo final (MultinomialNB) fue reentrenado con el 100% del corpus y serializado con `joblib`. El notebook `deployment.ipynb` carga el pipeline y clasifica nuevos correos preprocesando el texto con la misma función de limpieza utilizada durante el entrenamiento. Sobre un conjunto de 7 correos no vistos, el modelo clasificó correctamente todos los mensajes, asignando la etiqueta 0 (ham) o 1 (spam) según lo detectado por el modelo.

<a id="consideraciones-eticas"></a>
# 6. Consideraciones éticas

El desarrollo de sistemas de detección de spam implica consideraciones éticas que pueden tenerse en cuenta tanto en el diseño como en el despliegue del modelo.

En primer lugar, existe el riesgo de **falsos positivos**: correos legítimos clasificados incorrectamente como spam. Esto puede tener consecuencias graves, como la pérdida de comunicaciones importantes. Por esta razón, la métrica de recall fue priorizada durante el entrenamiento, minimizando los falsos negativos, aunque esto puede aumentar ligeramente los falsos positivos.

En segundo lugar, el corpus Enron utilizado para el entrenamiento proviene de correos reales de empleados de una empresa específica, lo que puede introducir **sesgos culturales y lingüísticos**. Un modelo entrenado exclusivamente con este corpus podría tener un desempeño menor en poblaciones o contextos diferentes, como correos en otros idiomas o de otros sectores.

En tercer lugar, la automatización del filtrado de correos implica un procesamiento masivo de **comunicaciones privadas**. Aunque en este trabajo los datos son públicos y anonimizados, en un entorno de producción real sería  necesario garantizar el cumplimiento de regulaciones de privacidad como el GDPR o la ley 1581 de Colombia.

Finalmente, a medida que los modelos de detección mejoran, los generadores de spam también evolucionan. El uso de modelos de lenguaje para generar spam más sofisticado representa un desafío ético y técnico emergente que los sistemas actuales, incluido el presentado en este trabajo, no están diseñados para enfrentar completamente. 


<a id="conclusiones"></a>
# 7. Conclusiones

Este trabajo evaluó el desempeño de tres variantes de Naive Bayes frente a un Perceptrón Multicapa en la tarea de detección de spam sobre el corpus Enron. A partir de los resultados obtenidos, se derivan las siguientes conclusiones:

En primer lugar, la hipótesis central del trabajo se confirma parcialmente: **una mayor complejidad del modelo no garantiza un mejor desempeño**. El MLP obtuvo el mejor balance general (F1=0.9930, Accuracy=0.9929), pero la diferencia frente a ComplementNB (F1=0.9903) es marginal, siendo el MLP considerablemente más costoso en tiempo de entrenamiento.

En segundo lugar, **MultinomialNB demostró ser el modelo más efectivo para minimizar falsos negativos**, alcanzando el recall más alto (0.9985). En el contexto de detección de spam, donde dejar pasar un correo malicioso tiene un costo mayor que filtrar uno legítimo, este resultado es especialmente relevante.

En tercer lugar, la **calidad del preprocesamiento tuvo un impacto determinante** en el rendimiento de todos los modelos. La combinación de lematización, eliminación de stop words y representación TF-IDF con bigramas permitió que incluso los modelos más simples alcanzaran métricas superiores al 98%.

Finalmente, se concluye que **Naive Bayes, en sus variantes Multinomial y Complement, representa una opción sólida y eficiente** para la detección de spam en producción, ofreciendo un desempeño con tiempos de entrenamiento e inferencia notablemente menores que los modelos de redes neuronales.

<a id="bibliografia"></a>
# 8. Bibliografía  

[1] C. Ellis and R. Brandl, "Spam Statistics 2026: Survey on Junk Email, AI Scams & Phishing," EmailTooltester, Oct. 16, 2024. [Online]. Available: https://www.emailtooltester.com/en/blog/spam-statistics/. [Accessed: Apr. 27, 2026].

[2] X. Wang, "Spam Filtering in the Modern Era: A Review of Machine Learning, Deep Learning, and System Comparisons," in *Proc. 2nd Int. Conf. Data Analysis and Machine Learning (DAML)*, 2024, pp. 451-458, doi: 10.5220/0013526000004619.

[3] S. Kaddoura, G. Chandrasekaran, D. E. Popescu, and J. H. Duraisamy, "A systematic literature review on spam content detection and classification," *PeerJ Computer Science*, vol. 8, e830, 2022, doi: 10.7717/peerj-cs.830.

[4] V. Metsis, I. Androutsopoulos, and G. Paliouras, "Spam Filtering with Naive Bayes - Which Naive Bayes?," in *Proc. 3rd Conf. on Email and Anti-Spam (CEAS)*, 2006. [Online]. Available: https://www.researchgate.net/publication/221650814_Spam_Filtering_with_Naive_Bayes_-_Which_Naive_Bayes. [Accessed: Apr. 27, 2026].

[5] S. A. Mohammed et al., "Classifying Unsolicited Bulk Email (UBE) using Python Machine Learning Techniques," 2013. [Online]. Available: https://www.researchgate.net/publication/236970412_Classifying_Unsolicited_Bulk_Email_UBE_using_Python_Machine_Learning_Techniques. [Accessed: Apr. 27, 2026].

[6] A. S. Hammad and A. M. El-Halees, "An Approach for Detecting Spam in Arabic Opinion Reviews," 2013. [Online]. Available: https://www.researchgate.net/publication/262765511_An_Approach_for_Detecting_Spam_in_Arabic_Opinion_Reviews. [Accessed: Apr. 27, 2026].

[7] M. Fusilier et al., "Detection of Opinion Spam with Character n-grams," 2015. [Online]. Available: https://www.researchgate.net/publication/312829622_Detection_of_Opinion_Spam_with_Character_n-grams. [Accessed: Apr. 27, 2026].

[8] J. Wu et al., "Spam detection study (Twitter dataset)," 2017. [Online]. Available: https://doi.org/10.1145/3014812.3014815. [Accessed: Apr. 27, 2026].

[9] R. Aiyar and N. Shetty, "Character n-gram based spam detection study," 2018. [Online]. Available: https://www.sciencedirect.com/science/article/pii/S1877050918309153. [Accessed: Apr. 27, 2026].

[10] S. Mani, S. Kumari, A. Jain, and P. Kumar, "Spam Review Detection Using Ensemble Machine Learning," in *Machine Learning and Data Mining in Pattern Recognition (MLDM 2018)*, Lecture Notes in Computer Science, vol. 10935. Cham, Switzerland: Springer, 2018, pp. 198-209, doi: 10.1007/978-3-319-96133-0_15.

[11] R. M. K. Saeed, S. Rady, and T. F. Gharib, "An ensemble approach for spam detection in Arabic opinion texts," *Journal of King Saud University - Computer and Information Sciences*, vol. 34, no. 1, pp. 1407-1416, 2022, doi: 10.1016/j.jksuci.2019.10.002.

[12] O. Kontsewaya et al., "Spam detection in email using machine learning techniques," 2021. [Online]. Available: https://www.sciencedirect.com/science/article/pii/S1877050921013016. [Accessed: Apr. 27, 2026].
