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
---

<!-- _class: lead -->

# Detección de spam en correo electrónico
## Naive Bayes frente a perceptrón multicapa (MLP)

**Santiago Puerta** · santiago.puertaf@upb.edu.co  
**Iker Acevedo Vargas** · iker.acevedo@upb.edu.co

---

## Contenido

1. Descripción del problema  
2. Estado del arte  
3. Metodología y experimento  
4. Resultados  
5. Conclusiones  

---

<!-- _header: '' -->

# 1. Descripción del problema

---

## Contexto

- Se estima que **cada día** se envían **160 mil millones** de correos electrónicos no deseados<sup class="cite-mark">[1]</sup>.

- Las **instituciones financieras** concentran cerca del **27 %** de los mensajes fraudulentos (objetivos frecuentes de atacantes)<sup class="cite-mark">[1]</sup>.

- En **servicios de entrega**, alrededor de **1.100 millones** de mensajes fraudulentos estuvieron relacionados con este tema, con impacto en confianza y reputación<sup class="cite-mark">[1]</sup>.

<small class="cita-biblio"><strong>[1]</strong> C. Ellis y R. Brandl, «Spam Statistics 2026: Survey on Junk Email, AI Scams y Phishing», EmailTooltester, oct. 2024.</small>

---

<!-- _header: '' -->

# 2. Estado del arte

---

## Panorama

- Desde **reglas heurísticas** y **NB / SVM** hasta **embeddings**, **MLP** y **modelos de lenguaje**.
- **Naive Bayes** sigue siendo referencia en texto por **simplicidad**, **velocidad** y buen rendimiento en muchos corpus.
- La literatura discute **representación** (palabras vs n-gramas de caracteres), **variantes de NB** y **trade-off** precisión vs coste.

---

## Trabajos representativos (I)

| Referencia | Idea clave |
|------------|------------|
| **Metsis et al. (2006)** | Comparación de variantes de NB en datos tipo Enron-Spam; la **variante** y la **binarización** importan. |
| **Mohammed et al. (2013)** | Léxico dinámico + varios clasificadores; NB y SVM destacan; léxicos **evolucionan**. |
| **Fusilier et al. (2015)** | **N-gramas de caracteres**; Multinomial NB competitivo; mejora frente a n-gramas de palabras en su setting. |

---

## Trabajos representativos (II)

| Referencia | Idea clave |
|------------|------------|
| **Wu et al. (2017)** | Representaciones Word2Vec/Doc2Vec + modelos profundos; **MLP** por encima de NB en Twitter; NB sensible al **desbalance**. |
| **Aiyar & Shetty (2018)** | N-gramas de caracteres; **SVM** puede superar a NB al crecer $n$. |
| **Kontsewaya et al. (2021)** | Correo en inglés (Kaggle); **NB y regresión logística** muy altos; palabras **OOV** limitan NB. |

---

## Síntesis

- **NB** sigue siendo **baseline fuerte** en spam y texto corto.
- Modelos **más expresivos** pueden ganar según **corpus**, **desbalance** y **representación**.
- **Wang (2024):** revisión moderna — NB eficiente; modelos profundos captan más **contexto**; retos: *concept drift*, spam asistido por LLM.

→ Motiva una comparación **controlada** NB vs MLP en **un mismo pipeline** (este proyecto).

---

## Pregunta de investigación

En el estado del arte se han evaluado, entre otros, **Naive Bayes** (variantes), **SVM**, **bosques / ensambles**, **MLP** y representaciones más ricas (**Word2Vec**, **Doc2Vec**, modelos de lenguaje) — con distintos **trade-offs** entre precisión y coste.

**¿Un modelo más complejo (MLP) ofrece mejor desempeño que una familia de modelos simples (Naive Bayes)?**

- Se contrasta no solo la **calidad de predicción**, sino también el **coste computacional** y la **aplicabilidad** (curvas de aprendizaje, tiempos de entrenamiento e inferencia).

---

<!-- _header: '' -->

# 3. Metodología y experimento

---

## Flujo general

1. Carga de datos y **reserva** de una fracción mínima para **despliegue** (correos no vistos en entrenamiento).
2. **Preprocesamiento:** limpieza de cabeceras y ruido típico de correo, tokenización, *stopwords*, lematización (NLTK).
3. **Representación:** n-gramas `(1, 2)` con **TF-IDF** o **conteos** según el modelo.
4. **Modelos:** variantes de **Naive Bayes** + **MLPClassifier**; búsqueda de hiperparámetros.
5. **Evaluación:** métricas de clasificación, comparación, curvas ROC, curvas de aprendizaje, tiempos.

---

## Exploración de datos (EDA)

### ¿Existe una diferencia clara entre correo HAM y SPAM?

Antes de ajustar clasificadores, revisamos si el texto muestra **patrones distintivos** por clase (exploración **cualitativa**, no sustituye al modelo).

---

## Nubes de palabras (HAM vs SPAM)

![width:900px](../img/words-cloud-spam-vs-ham.png)

- **Spam:** suelen dominar términos asociados a promociones, ofertas y formulaciones repetitivas.
- **HAM (Enron):** vocabulario **corporativo** (nombres, dominios de negocio, hilos internos).
- **Matiz:** hay **solapamiento**; la nube resume **frecuencia / peso agregado**, no prueba estadística.

---

## Longitud del texto (HAM vs SPAM)

![width:900px](../img/boxplot-spam-vs-ham.png)

- Comparación de **longitud** (caracteres o número de palabras tras limpieza).
- Sirve para ver **medianas**, **dispersión** y **outliers** por clase.
- Una clase más larga o más variable **no implica** por sí sola separabilidad lineal en el espacio de características.

---

## Modelos Naive Bayes considerados

![width:720px](../img/naive-bayes-algorithms-comparison.png)

- **Multinomial** (TF-IDF y conteos).
- **Multinomial con vectores binarios** (`CountVectorizer(binary=True)`).
- **Bernoulli** y **Complement** NB (según configuración del notebook).
- **MLP:** red *feed-forward* con ajuste de arquitectura e hiperparámetros.

---

## Entrenamiento y evaluación

- **Partición:** `train_test_split` **estratificado** (p. ej. 80 % / 20 %), semilla fija.
- **Métricas:** exactitud, precisión, exhaustividad, F1-score; matrices de confusión y **ROC-AUC** donde aplique.
- **Selección:** comparación sistemática entre pipelines y búsqueda en rejilla / aleatoria según celda del notebook.

---

<!-- _header: '' -->

# 4. Resultados

---

## Curvas ROC

![width:880px](../img/models-roc-curves.png)

Comparación visual del trade-off **tasa de verdaderos positivos / falsos positivos** entre modelos ajustados.

---

## Curvas de aprendizaje

![width:880px](../img/models-learning-curve.png)

- Muestra cómo evoluciona el rendimiento al **aumentar el tamaño del conjunto de entrenamiento**.
- Ayuda a discutir **varianza**, **sesgo** y si más datos seguirían mejorando cada familia de modelo.

---

## Coste computacional

![width:880px](../img/models-fit-vs-inference.png)

- **Tiempo de ajuste** frente a **tiempo de inferencia** (o barras comparables según la figura exportada).
- Complementa la precisión: un modelo ligeramente mejor pero **mucho más caro** puede ser menos atractivo en producción.

---

<!-- _header: '' -->

# 5. Conclusiones

---

## Mensajes clave

- En detección de spam sobre **Enron**, los modelos evaluados alcanzan **métricas altas**; la comparación NB vs MLP debe leerse junto con las **figuras** de esta presentación y el notebook.
- **Naive Bayes** mantiene el atractivo de **simplicidad y rapidez**; el **MLP** puede aportar ganancias según hiperparámetros y representación.
- El **EDA** sugiere **diferencias léxicas y de longitud** entre HAM y SPAM, con matices de solapamiento — el paso riguroso es la **evaluación** bajo el mismo protocolo.

---

## Trabajo futuro y ética

- Incorporar **concept drift**, datos más recientes y **spam generado** con modelos de lenguaje.
- Considerar **sesgos** del corpus (Enron no representa todo el mundo), **privacidad** y **transparencia** en despliegue.

---

<!-- _class: lead -->

# Gracias

## Preguntas

Notebook: `notebooks/nb_vs_mlp.ipynb` · Documentación: `README.md`
