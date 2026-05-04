---
marp: true
theme: default
size: 16:9
paginate: true
header: Detección de spam — Naive Bayes vs MLP
style: |
  section { font-size: 28px; }
  h1 { font-size: 40px; }
  h2 { font-size: 34px; }
  img { max-height: 420px; }
  sup.cite-mark { font-size: 0.55em; font-weight: 600; vertical-align: super; line-height: 0; }
  small.cita-biblio { font-size: 0.48em; line-height: 1.3; color: #444; display: block; margin-top: 0.35em; }
  table { font-size: 0.78em; }
---
<!-- _class: lead -->

# Detección de spam en correo electrónico
## Naive Bayes frente a perceptrón multicapa (MLP)

**Santiago Puerta** · santiago.puertaf@upb.edu.co  
**Iker Acevedo Vargas** · iker.acevedo@upb.edu.co

---
<!-- _footer: 'Santiago' -->
## Contenido

1. Descripción del problema  
2. Estado del arte  
3. Metodología y experimento  
4. Resultados  
5. Conclusiones  

---
<!-- _footer: 'Santiago' -->
<!-- _header: '' -->

# 1. Descripción del problema

---
<!-- _footer: 'Santiago' -->
## Contexto

- Se estima que **cada día** se envían **160 mil millones** de correos electrónicos no deseados<sup class="cite-mark">[1]</sup>.

- Las **instituciones financieras** concentran cerca del **27 %** de los mensajes fraudulentos (objetivos frecuentes de atacantes)<sup class="cite-mark">[1]</sup>.

- En **servicios de entrega**, alrededor de **1.100 millones** de mensajes fraudulentos estuvieron relacionados con este tema, con impacto en confianza y reputación<sup class="cite-mark">[1]</sup>.

<small class="cita-biblio"><strong>[1]</strong> C. Ellis y R. Brandl, «Spam Statistics 2026: Survey on Junk Email, AI Scams y Phishing», EmailTooltester, oct. 2024.</small>

---
<!-- _footer: 'Santiago' -->
<!-- _header: '' -->

# 2. Estado del arte

---
<!-- _footer: 'Santiago' -->
## Panorama

- El spam ha impulsado la evolución de detectores: desde **reglas heurísticas** y **NB / SVM** hasta **embeddings**, **MLP** y **modelos de lenguaje** como BERT<sup class="cite-mark">[2,3]</sup>.
- **Naive Bayes** sigue siendo referencia en texto por **simplicidad**, **velocidad** y buen rendimiento en muchos corpus<sup class="cite-mark">[2]</sup>.
- La literatura discute **representación** (palabras vs n-gramas de caracteres), **variantes de NB** y **trade-off** precisión vs coste.

<small class="cita-biblio"><strong>[2]</strong> X. Wang, «Spam Filtering in the Modern Era: A Review of Machine Learning, Deep Learning, and System Comparisons», en <em>Proc. 2nd Int. Conf. Data Analysis and Machine Learning (DAML)</em>, 2024, pp. 451-458, doi: 10.5220/0013526000004619.<br><strong>[3]</strong> S. Kaddoura et al., «A systematic literature review on spam content detection and classification», <em>PeerJ Computer Science</em>, vol. 8, e830, 2022, doi: 10.7717/peerj-cs.830.</small>

---
<!-- _footer: 'Iker' -->
## Trabajos representativos (I)

| Referencia | Conjunto de datos | Idea clave | Mejor modelo |
|------------|-------------------|------------|--------------|
| **Metsis et al. (2006)** | Enron-Spam (6 subconjuntos, ~5 000–6 000 mensajes c/u) | Comparación de variantes de NB; la **variante** y la **binarización** importan. | **Multinomial con atributos booleanos** — mejor balance precisión/eficiencia. |
| **Mohammed et al. (2013)** | Email-1431 (544 correos balanceados) | Léxico dinámico + varios clasificadores; léxicos **evolucionan**. | **NB y SVM** empatados como los más efectivos. |
| **Fusilier et al. (2015)** | 1 600 reseñas de hoteles | **N-gramas de caracteres**; mejora frente a n-gramas de palabras entre 2.1–2.3 %. | **Multinomial NB** con n-gramas de caracteres. |

---
<!-- _footer: 'Iker' -->
## Trabajos representativos (II)

| Referencia | Conjunto de datos | Idea clave | Mejor modelo |
|------------|-------------------|------------|--------------|
| **Hammad & El-Halees (2013)** | Reseñas de hoteles en árabe | Características de contenido, metadatos y atributos del autor. | **Naive Bayes** — 99.2 % de precisión. |
| **Wu et al. (2017)** | Twitter (+2 M de mensajes) | Word2Vec/Doc2Vec + modelos profundos; NB sensible al **desbalance**. | **MLP** — ~95 %, superando a NB (80–85 %). |
| **Aiyar & Shetty (2018)** | 13 000 comentarios de YouTube | N-gramas de caracteres; NB el más bajo al crecer $n$. | **SVM** con 6-gramas de caracteres. |

---
<!-- _footer: 'Santiago' -->
## Trabajos representativos (III)

| Referencia | Conjunto de datos | Idea clave | Mejor modelo |
|------------|-------------------|------------|--------------|
| **Mani et al. (2018)** | Reseñas en inglés | Ensamble NB + RF + SVM compensa limitaciones individuales. | **Ensamble (NB + RF + SVM)** — 87.68 % de precisión. |
| **Saeed et al. (2019)** | 2 corpus de reseñas de hoteles en árabe | Stacking + módulo de negaciones; NB destacado dentro del ensamble. | **Stacking** — >95 % de precisión. |
| **Kontsewaya et al. (2021)** | 5 728 correos en inglés (Kaggle) | NB y regresión logística muy altos; palabras **OOV**<sup class="cite-mark">†</sup> limitan NB. | **Regresión Logística y NB** — ~99 % de precisión. |

<small class="cita-biblio"><strong>†</strong> <em>OOV (Out-Of-Vocabulary):</em> términos que no aparecieron en el conjunto de entrenamiento y que el modelo no puede representar directamente, lo que puede afectar negativamente su probabilidad estimada.</small>

---
<!-- _footer: 'Iker' -->
## Síntesis

- **NB** sigue siendo *baseline* fuerte no solo en spam de correo electrónico, sino también en detección de spam en **reseñas, redes sociales y foros** (YouTube, Twitter, hoteles) — lo que evidencia su versatilidad más allá del correo.
- Modelos **más expresivos** pueden ganar según **corpus**, **desbalance** y **representación**.
- **Wang (2024):** revisión moderna — NB eficiente; modelos profundos captan más **contexto**; retos: *concept drift*, spam asistido por LLM.

→ Motiva una comparación **controlada** NB vs MLP en **un mismo pipeline** (este proyecto).

---
<!-- _footer: 'Santiago' -->
## Pregunta de investigación

En el estado del arte se han evaluado, entre otros, **Naive Bayes** (variantes), **SVM**, **bosques / ensambles**, **MLP** y representaciones más ricas (**Word2Vec**, **Doc2Vec**, modelos de lenguaje) — con distintos **trade-offs** entre precisión y coste.

**¿Un modelo más complejo (MLP) ofrece mejor desempeño que una familia de modelos simples (Naive Bayes)?**

- Se contrasta no solo la **calidad de predicción**, sino también el **coste computacional** y la **aplicabilidad** (curvas de aprendizaje, tiempos de entrenamiento e inferencia).

---
<!-- _footer: 'Santiago' -->
<!-- _header: '' -->

## ¿Qué es Naive Bayes?

Una **familia** de clasificadores probabilísticos basados en el **Teorema de Bayes** con la suposición de **independencia condicional** entre características:

$$P(C \mid X) = \frac{P(X \mid C)\,P(C)}{P(X)}$$

**¿Por qué funciona tan bien para spam?**

- El spam suele estar dominado por **patrones léxicos repetitivos** (palabras clave, frases promocionales) que se detectan eficientemente con probabilidades por término.
- **Baja complejidad computacional** — ideal para grandes volúmenes de correos.
- **Buena generalización** con pocos datos de entrenamiento gracias al *prior* bayesiano.
- Robusto ante vocabularios grandes y dispersos (característica típica del texto).

---
<!-- _footer: 'Santiago' -->
## Bernoulli NB — Presencia o ausencia

Representa cada mensaje como un **vector binario**: 1 si la palabra está, 0 si no.

$$P(C \mid X) = \frac{\left[\prod_{i=1}^{n} P(x_i \mid C)^{x_i}(1-P(x_i \mid C))^{1-x_i}\right] \cdot P(C)}{P(X)}$$

$$P(X) = \prod_{i=1}^{n} P(x_i \mid \text{Spam})^{x_i}(1-P(x_i \mid \text{Spam}))^{1-x_i} \cdot P(\text{Spam})$$
$$+ \prod_{i=1}^{n} P(x_i \mid \text{Ham})^{x_i}(1-P(x_i \mid \text{Ham}))^{1-x_i} \cdot P(\text{Ham})$$

> **Caso de uso típico:** mensajes cortos o SMS donde la presencia de ciertas palabras ya es señal suficiente.

<small class="cita-biblio"><strong>†</strong> El factor $(1 - P(x_i \mid C))$ representa la probabilidad de que la palabra <em>esté ausente</em> dado que el mensaje pertenece a la clase $C$. Cuando $x_i = 0$ este factor se activa, penalizando o favoreciendo la clase según cuán frecuente sea ese término en ella.</small>

---
<!-- _footer: 'Santiago' -->
## Bernoulli NB — Limitaciones

| ⚠️ Limitación | Explicación |
|--------------|-------------|
| **Ignora la frecuencia** | "Premio Premio Premio" se trata igual que "Premio" — el modelo no distingue cuántas veces aparece un término, perdiendo una señal discriminativa clave en spam. |
| **Evalúa todo el vocabulario** | Un mensaje de 2 palabras con vocabulario de 5 000 requiere un vector de 5 000 posiciones (4 998 ceros). Además, las probabilidades pequeñas de los términos **ausentes** van diluyendo el producto del likelihood — análogo al **desvanecimiento del gradiente**. |
| **Sensible al tamaño del vocabulario** | A mayor vocabulario, más términos ausentes participan en el cálculo del evidence, amplificando el efecto de dilución y pudiendo sesgar la clasificación hacia la clase con mayor prior. |

---
<!-- _footer: 'Santiago' -->
## Bernoulli NB — Caso práctico

**Escenario:** 80 HAM / 20 SPAM · Vocabulario: {premio, ganaste, gratis, dinero, oferta} · Mensaje: *"Premio Ganaste"* → vector $(1,1,0,0,0)$

**Posterior:**

$$P(\text{Spam} \mid \text{mensaje}) = \frac{[P(\text{premio} \mid \text{Sp}) \cdot P(\text{ganaste} \mid \text{Sp}) \cdot (1 - P(\text{gratis} \mid \text{Sp})) \cdot (1 - P(\text{dinero} \mid \text{Sp})) \cdot (1 - P(\text{oferta} \mid \text{Sp}))] \cdot P(\text{Sp})}{P(X)}$$

**Evidence:**

$$P(X) = [P(\text{premio} \mid \text{Sp}) \cdot P(\text{ganaste} \mid \text{Sp}) \cdot (1{-}P(\text{gratis} \mid \text{Sp})) \cdot (1{-}P(\text{dinero} \mid \text{Sp})) \cdot (1{-}P(\text{oferta} \mid \text{Sp})) \cdot P(\text{Sp})]$$
$$+\ [P(\text{premio} \mid \text{Ham}) \cdot P(\text{ganaste} \mid \text{Ham}) \cdot (1{-}P(\text{gratis} \mid \text{Ham})) \cdot (1{-}P(\text{dinero} \mid \text{Ham})) \cdot (1{-}P(\text{oferta} \mid \text{Ham})) \cdot P(\text{Ham})]$$

<small class="cita-biblio"><strong>†</strong> Los 3 términos ausentes (gratis, dinero, oferta) participan activamente mediante los factores $(1 - P(x_i \mid C))$ tanto en el <strong>likelihood</strong> del numerador como en el cálculo del <strong>evidence</strong>, diluyendo la probabilidad final en ambos componentes.</small>

---
<!-- _footer: 'Santiago' -->
## Multinomial NB — Frecuencia de términos

Representa el mensaje con **conteos**: captura cuántas veces aparece cada palabra.

$$P(C \mid X) = \frac{\left[\prod_{i=1}^{n} P(x_i \mid C)^{f_i}\right] \cdot P(C)}{P(X)}$$

$$P(X) = \prod_{i=1}^{n} P(x_i \mid \text{Spam})^{f_i} \cdot P(\text{Spam}) + \prod_{i=1}^{n} P(x_i \mid \text{Ham})^{f_i} \cdot P(\text{Ham})$$

donde $f_i$ es la frecuencia del término $x_i$ en el mensaje. A diferencia de Bernoulli, los términos con $f_i = 0$ contribuyen con $P(x_i \mid C)^0 = 1$, por lo que **las palabras ausentes no participan en el cálculo** — ni en el likelihood ni en el evidence.

> **Caso de uso típico:** correos donde la repetición de palabras clave ("gratis", "oferta", "premio") es indicativa de spam.

---
<!-- _footer: 'Santiago' -->
## Multinomial NB — Limitaciones

| ⚠️ Limitación | Explicación |
|--------------|-------------|
| **Independencia condicional** | Sigue ignorando la coocurrencia entre palabras; "Premio Ganaste" se trata como dos eventos independientes, lo que puede ser simplista en spam más elaborado. |
| **Sensible a la longitud del texto** | La relevancia de un término que aparece múltiples veces se diluye en textos largos — su influencia relativa decrece entre el resto de términos presentes. |
| **Concept drift** | Su rendimiento no mejora proporcionalmente con más datos si la distribución del spam cambia; no captura eficientemente variaciones drásticas en la redacción de mensajes. |

---
<!-- _footer: 'Santiago' -->
## Multinomial NB — Caso práctico

**Escenario:** 80 HAM / 20 SPAM · Vocabulario: {premio, ganaste, gratis, dinero, oferta} · Mensaje: *"Premio Premio Ganaste"* → vector $(2,1,0,0,0)$

**Posterior:**

$$P(\text{Spam} \mid \text{mensaje}) = \frac{[P(\text{premio} \mid \text{Sp})^2 \cdot P(\text{ganaste} \mid \text{Sp})^1] \cdot P(\text{Sp})}{P(X)}$$

**Evidence:**

$$P(X) = [P(\text{premio} \mid \text{Sp})^2 \cdot P(\text{ganaste} \mid \text{Sp}) \cdot P(\text{Sp})]$$
$$+\ [P(\text{premio} \mid \text{Ham})^2 \cdot P(\text{ganaste} \mid \text{Ham}) \cdot P(\text{Ham})]$$

<small class="cita-biblio"><strong>†</strong> Nótese que gratis, dinero y oferta ($f_i = 0$) no aparecen en ninguna parte del cálculo — a diferencia de Bernoulli, su ausencia no penaliza ni sesga el resultado. Además, la repetición de "premio" ($f_i = 2$) amplifica su influencia en el posterior.</small>

---
<!-- _footer: 'Santiago' -->
## Bernoulli vs Multinomial — Trade-offs

| Dimensión | Bernoulli | Multinomial |
|-----------|-----------|-------------|
| **Representación** | Binaria (presencia/ausencia) | Conteos o TF-IDF |
| **Términos ausentes** | Penaliza explícitamente | Los ignora |
| **Longitud del texto** | Peor para textos largos | Mejor para textos largos |
| **Desempeño en email** | Competitivo (palabras clave únicas) | Generalmente superior (frecuencia importa) |

> En la literatura (Metsis et al., 2006), **Multinomial con binarización** (*boolean*) suele lograr el mejor balance en corpus de correo electrónico.

---
<!-- _footer: 'Iker' -->
<!-- _header: '' -->

# 3. Metodología y experimento

---
<!-- _footer: 'Iker' -->
## Conjunto de datos — Enron-Spam

- **Fuente:** corpus público derivado de los correos internos de Enron, ampliamente usado como benchmark en detección de spam.
- **Composición:** mensajes etiquetados como **HAM** (correo legítimo corporativo) y **SPAM** (mensajes no deseados añadidos artificialmente).
- **Características:**
  - Vocabulario **corporativo y técnico** en los mensajes HAM (nombres, dominios, hilos internos).
  - Amplia variedad de spam: ofertas, phishing, boletines no solicitados.
- **Consideración:** el corpus no representa todos los escenarios del mundo real (sesgo hacia inglés corporativo); esto se discute en las conclusiones.

---
<!-- _footer: 'Iker' -->
## Flujo general

1. Carga de datos y **reserva** de una fracción mínima para **despliegue** (≈ 0.02%).
2. **Preprocesamiento:** limpieza de cabeceras y ruido típico de correo, tokenización, *stopwords*, lematización (NLTK).
3. **Representación:** n-gramas `(1, 2)` con **TF-IDF** o **aparición binaria** según el modelo.
4. **Modelos:** variantes de **Naive Bayes** y **MLPClassifier**; búsqueda de hiperparámetros.
5. **Evaluación:** métricas de clasificación, comparación, curvas ROC, curvas de aprendizaje, tiempos.

---
<!-- _footer: 'Iker' -->
## Exploración de datos (EDA)

### ¿Existe una diferencia clara entre correo HAM y SPAM?

Antes de ajustar clasificadores, revisamos si el texto muestra **patrones distintivos** por clase.

---
<!-- _footer: 'Iker' -->
## Nubes de palabras (HAM vs SPAM)

![width:900px](../img/words-cloud-spam-vs-ham.png)

---
<!-- _footer: 'Iker' -->
## Nubes de palabras — Interpretación

- **Spam:** predominan `www`, `http` y `com` en 1 y 2-gramas — la mayoría de correos spam traen **hipervínculos asociados**, señal discriminativa fuerte. Acompañados de vocabulario de captación: `free`, `offer`, `save`, `money`.
- **HAM (Enron):** vocabulario **corporativo muy específico** — `enron` — y comunicación interna: `schedule`, `meeting`, `attached`.
---
<!-- _footer: 'Iker' -->
## Longitud del texto (HAM vs SPAM)

![width:900px](../img/boxplot-spam-vs-ham.png)

---
<!-- _footer: 'Iker' -->
## Longitud del texto — Interpretación

- **SPAM:** correos generalmente más cortos — caja más compacta y bigote superior menor tanto en caracteres como en palabras. Sin embargo, es la clase con los **outliers más extremos** (~20 000 caracteres), lo que refleja una **alta variabilidad**.
- **HAM:** correos típicamente más largos — caja más ancha y mayor distancia entre P50 y P75, con bigote superior más extenso (~3 500 palabras, ~3 500 caracteres). También presenta abundantes outliers de correos muy extensos.
- **Conclusión:** el enorme solapamiento entre clases y la abundancia de outliers en ambas indica que la **longitud sola no es un discriminador confiable** entre HAM y SPAM.

---
<!-- _footer: 'Iker' -->
## ¿Por qué no basta con heurísticas?

El EDA revela que construir reglas simples para separar HAM y SPAM es prácticamente inviable:

- **Longitud:** las distribuciones se solapan casi por completo — un umbral de caracteres o palabras clasificaría mal una fracción inaceptable de correos.
- **Palabras clave:** términos como `subject`, `would` o `please` aparecen en ambas clases — ninguna palabra por sí sola es discriminadora confiable.

→ Se requiere una técnica que evalúe **combinaciones de señales simultáneamente** y de forma **probabilística o paramétrica** — motivando la comparación **Naive Bayes vs MLP**.

---
<!-- _footer: 'Santiago' -->
## Modelos Naive Bayes considerados

![width:1200px](../img/naive-bayes-algorithms-comparison.png)

---
<!-- _footer: 'Iker' -->
## Entrenamiento y evaluación

- **Partición:** `train_test_split` **estratificado** (p. ej. 80 % / 20 %), semilla fija.
- **Métricas:** exactitud, precisión, exhaustividad, F1-score; matrices de confusión y **ROC-AUC** donde aplique.
- **Selección:** comparación sistemática entre pipelines y búsqueda de mejores hiperparametros. 

---
<!-- _footer: 'Santiago' -->
<!-- _header: '' -->

# 4. Resultados

---
<!-- _footer: 'Santiago' -->
## Curvas ROC

![width:800px](../img/models-roc-curves.png)

---
<!-- _footer: 'Santiago' -->
## Curvas ROC — Interpretación

- **MLP** lidera con la mejor curva TP vs FP a tasas de FP bajas (0–0.005), pero comparte **AUC = 0.999** con MultinomialNB y ComplementNB — la diferencia es de décimas de punto porcentual, prácticamente **despreciable**.
- **MultinomialBinaryNB** es el único rezagado con AUC = 0.998 — una diferencia de 0.001, igualmente marginal.
- Con 3 de 4 modelos en AUC = 0.999, el **criterio ROC no es suficiente para elegir entre ellos** — la decisión debe apoyarse en otros factores como coste computacional o curvas de aprendizaje.


---
<!-- _footer: 'Santiago' -->
## Curvas de aprendizaje

![width:1200px](../img/models-learning-curve.png)

---
<!-- _footer: 'Santiago' -->
## Curvas de aprendizaje — Interpretación

- **MultinomialNB** es el modelo más estable: training y CV convergen cerca de 0.998, con la **menor desviación estándar** de los cuatro — predicciones consistentes ante datos nuevos, característica clave en entornos productivos.
- **MLP** muestra una caída progresiva en CV conforme crecen los datos de entrenamiento, con una **banda de incertidumbre amplia** — no es overfitting (la diferencia porcentual es mínima), pero sí una señal de menor estabilidad ante datos desconocidos; factor a considerar al elegir modelo, especialmente contrastando con su AUC = 0.999.
- **MultinomialBinaryNB y ComplementNB** presentan las mayores brechas entre training y CV, con bandas de CV muy amplias — mayor varianza y menor confiabilidad en producción.

---
<!-- _footer: 'Santiago' -->
## Coste computacional

![width:1200px](../img/models-fit-vs-inference.png)

---
<!-- _footer: 'Santiago' -->
## Coste computacional — Interpretación

- **Fit time:** MLP opera un **orden de magnitud por encima** de los modelos NB — si NB tarda 100 minutos en entrenarse, MLP tardaría ~4.000 minutos (~2.5 días).
- **Score time:** MLP es ~×3 más lento que sus contrapartes NB en inferencia — si NB tarda 1h en predecir, MLP tardaría ~3h. Además, su **banda de incertidumbre es la más amplia** de los cuatro modelos, lo que implica un comportamiento impredecible en entornos productivos.
- **NB:** tiempos de entrenamiento e inferencia notablemente inferiores, con bandas de desviación estándar estrechas — comportamiento **estable y predecible** a escala.
- **Conclusión:** un modelo con rendimiento marginalmente superior en AUC pero con un coste computacional ordenes de magnitud mayor, **no es necesariamente la mejor elección** en producción.

---
<!-- _footer: 'Iker' -->
# 5. Conclusiones

---
<!-- _footer: 'Iker' -->
## Mensajes clave

- **Mayor complejidad ≠ mejor desempeño:** MLP logra el mejor F1 (0.9930) pero la diferencia frente a ComplementNB (F1=0.9903) es marginal — a un coste computacional órdenes de magnitud mayor.
- **MultinomialNB minimiza falsos negativos** con el recall más alto (0.9985) — en detección de spam, dejar pasar un correo malicioso tiene un coste mayor que filtrar uno legítimo; este es el criterio prioritario.
- **El preprocesamiento es determinante:** lematización + stop words + TF-IDF con bigramas permitió que incluso los modelos más simples superaran el 98% en todas las métricas.
- **Veredicto:** Naive Bayes (Multinomial y Complement) es la opción más sólida para producción — desempeño competitivo, tiempos de entrenamiento e inferencia notablemente menores y comportamiento estable y predecible.

---
<!-- _footer: 'Iker' -->
## Trabajo futuro

- Siguiendo a Fusilier et al. (2015) y Aiyar & Shetty (2018), explorar **n-gramas de caracteres** en lugar de palabras para evaluar si la representación a nivel de letra mejora el rendimiento.
- Incorporar **concept drift**, datos más recientes y **spam generado** con modelos de lenguaje.
- Considerar **sesgos** del corpus (Enron no representa todo el mundo), **privacidad** y **transparencia** en despliegue.