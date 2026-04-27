# Resumen
Resumen de 5 lineas. 
# Introducción
El crecimiento exponencial de la comunicación digital ha traído consigo un aumento proporcional en la cantidad de mensajes no deseados (spam); se estima que “cada día se envían 160 mil millones de correos electrónicos no deseados” [1]. Asimismo, un agravante de esta situación se presenta tanto en instituciones financieras como en servicios de entrega. Las primeras, al ser objetivos frecuentes de los atacantes, llegan a concentrar “el 27% de los mensajes fraudulentos” [1]. En el caso de los servicios de entrega, el problema es diferente: alrededor de 1.100 millones de mensajes fraudulentos estuvieron relacionados con este tema [1], lo que genera una pérdida de confianza por parte de los usuarios y perjudica la reputación de estas empresas de cara a potenciales usuarios futuros.

Diversos enfoques han sido propuestos para abordar este problema, abarcando desde reglas heurísticas [2] y modelos clásicos de aprendizaje automático como Naive Bayes y Support Vector Machines [2], hasta arquitecturas más complejas basadas en redes neuronales, como Multi Layer Perceptron y modelos de lenguaje como BERT [3]. Sin embargo, a pesar de esta diversidad de enfoques y niveles de complejidad, surge una pregunta fundamental: ¿implica un mayor nivel de complejidad en el modelo un mejor rendimiento?

El presente artículo busca dar respuesta a esta pregunta mediante la comparación del desempeño de una familia de modelos simples, como lo son los Naive Bayes, frente a un modelo más complejo como Multi Layer Perceptron, aplicados a la tarea de detección de spam. Se analizará no solo la calidad de las predicciones, sino también las implicaciones en términos de eficiencia y aplicabilidad en escenarios reales.

# Contexto
Cuando hablamos de Naive Bayes, no nos referimos a un algoritmo en particular, sino, mas bien a una familia de algoritmos con diversas variantes: Bernoulli, Complement, Multinomial, entre otros, los cuales han sido ampliamente investigados y usados en la industria para la deteccion de spam.

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

## Bernoulli
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

## Multinomial
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

### Ejemplo práctico

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

# Estado del arte
# Resultados
# Consideraciones éticas
# Conclusiones
# Bibliografia  
[1] Spam Statistics 2026: New Data on Junk Email, AI Scams & Phishing https://www.emailtooltester.com/en/blog/spam-statistics/

[2] https://www.scitepress.org/Papers/2024/135260/135260.pdf

[3] https://pmc.ncbi.nlm.nih.gov/articles/PMC8802784/pdf/peerj-cs-08-830.pdf (*)

[4] https://www.researchgate.net/publication/221650814_Spam_Filtering_with_Naive_Bayes_-_Which_Naive_Bayes
