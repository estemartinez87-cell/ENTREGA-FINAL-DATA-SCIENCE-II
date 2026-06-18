# Predicción de riesgo epidemiológico en Argentina

Proyecto final de Machine Learning: clasificación binaria de "años de alto riesgo
epidemiológico" (reaparición de enfermedades prevenibles por vacunas) a partir de
cobertura de vacunación rezagada y contexto socioeconómico en Argentina (2000-2025).

## 1. Problema y objetivo

Argentina registró entre 2022 y 2025 una caída histórica en las coberturas de
vacunación (triple viral, antipoliomielítica) junto con un repunte de enfermedades
que estaban controladas (coqueluche, sarampión). Este proyecto formula esa situación
como un problema de **clasificación binaria supervisada**:

- **Variable objetivo (`riesgo_alto`):** 1 si el año tuvo casos confirmados de
  sarampión o casos de coqueluche por encima del percentil 75 histórico; 0 en caso
  contrario.
- **Métrica de éxito:** AUC-ROC (principal), con Accuracy, F1, Precision y Recall
  como métricas secundarias.
- **Variables predictoras:** cobertura de vacunación del año anterior (lag),
  pobreza, desempleo, inflación (transformada logarítmicamente), fecundidad
  adolescente y una variable de tendencia temporal.

El notebook completo está en [`notebooks/proyecto_final_ml.ipynb`](notebooks/proyecto_final_ml.ipynb).

## 2. Estructura del repositorio

```
proyecto_ml_argentina/
├── README.md                          ← este archivo
├── requirements.txt                   ← dependencias con versiones fijas
├── data/
│   ├── argentina_dataset.csv          ← dataset histórico nacional 2000-2025 (14 variables)
│   └── cobertura_provincial_2019_2023.csv  ← datos reales parciales por provincia
├── notebooks/
│   ├── proyecto_final_ml.py           ← código fuente principal (formato jupytext)
│   ├── proyecto_final_ml.ipynb        ← notebook principal ejecutado con outputs
│   ├── anexo_provincial.py            ← código fuente del anexo provincial
│   ├── anexo_provincial.ipynb         ← anexo provincial ejecutado con outputs
│   ├── prediccion_5_anios.py          ← código fuente: predicción a 5 años, 3 escenarios
│   ├── prediccion_5_anios.ipynb       ← predicción a 5 años ejecutada con outputs
│   ├── conclusion_final.py            ← código fuente de la conclusión narrativa final
│   └── conclusion_final.ipynb         ← conclusión narrativa que une los tres análisis
├── img/                                ← gráficos exportados (EDA, ROC, SHAP, ranking provincial, predicción 5 años, etc.)
└── extras/
    └── pipeline_vacunacion_reemergentes.py  ← análisis complementario de series de tiempo (ARIMA)
```

> **Nota sobre `notebooks/proyecto_final_ml.py`:** se mantiene en formato
> [Jupytext](https://jupytext.readthedocs.io/) (celdas marcadas con `# %%`) porque
> los archivos `.py` se versionan en git de forma mucho más legible que los `.ipynb`
> (que mezclan código con metadata JSON y outputs binarios). El `.ipynb` se regenera
> con el comando de la sección 4.

## 3. Fuentes de los datos

Todos los datos provienen de organismos oficiales argentinos e internacionales.
Los valores anteriores a 2015 en algunas series fueron reconstruidos por interpolación
a partir de publicaciones oficiales cuando no había series digitalizadas continuas;
los datos de 2015 en adelante están verificados contra la fuente primaria indicada.

| Variable | Fuente | Notas |
|---|---|---|
| Mortalidad infantil, esperanza de vida, fecundidad adolescente | DEIS (Ministerio de Salud), OPS/OMS Argentina | Serie 5 – Estadísticas Vitales |
| Cobertura triple viral, cobertura antipoliomielítica | Nomivac, Sociedad Argentina de Pediatría (SAP) | Coberturas nominalizadas |
| Casos de coqueluche y sarampión | Boletín Epidemiológico Nacional (BEN), SAP, OPS | Casos confirmados notificados |
| Pobreza, indigencia, desempleo | INDEC – Encuesta Permanente de Hogares (EPH) | Series semestrales/trimestrales anualizadas |
| Inflación (IPC), tipo de cambio | INDEC, BCRA | IPC oficial desde 2017; estimaciones previas |

Fuentes para actualizar los datos:
- https://datos.gob.ar (API de series de tiempo)
- https://www.argentina.gob.ar/salud/deis
- https://www.argentina.gob.ar/salud/epidemiologia/boletin
- https://www.indec.gob.ar

## 4. Cómo reproducir el proyecto

```bash
# 1. Clonar el repositorio
git clone <URL_DE_TU_REPO>
cd proyecto_ml_argentina

# 2. Crear un entorno virtual e instalar dependencias
python3 -m venv venv
source venv/bin/activate          # En Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Ejecutar el notebook de punta a punta y regenerar los outputs
cd notebooks
jupyter nbconvert --to notebook --execute --inplace proyecto_final_ml.ipynb

# 4. (Opcional) Abrir interactivamente
jupyter notebook proyecto_final_ml.ipynb
```

## 5. Resumen de resultados

| Modelo | AUC (CV 5-fold, tuning) | AUC (LOOCV, evaluación final) |
|---|---|---|
| Regresión Logística | 0.87 | **0.78** ← seleccionado |
| Random Forest | 0.80 | ver notebook |
| XGBoost | 0.73 | ver notebook |

El modelo final es una **Regresión Logística regularizada**, seleccionada por su
mejor AUC en evaluación Leave-One-Out (apropiada para el tamaño de muestra, n=25).
La variable con mayor poder explicativo según SHAP es la **cobertura de triple
viral del año anterior**, seguida de la inflación y la fecundidad adolescente.

### 5.1 Predicción a 5 años (2026-2030) — tres escenarios

Ver `notebooks/prediccion_5_anios.ipynb`. Extiende el modelo principal para
proyectar la probabilidad de un año de riesgo alto bajo tres trayectorias de
cobertura de vacunación (las variables socioeconómicas se mantienen
congeladas en su nivel de 2025 por la inestabilidad del ARIMA de inflación —
ver nota metodológica dentro del notebook):

| Año | Statu quo | Recuperación (cobertura → 70-72% en 2030) | Deterioro (-2pp/año) |
|---|---|---|---|
| 2026 | 85.8% | 85.8% | 85.8% |
| 2027 | 85.1% | 85.1% | 85.1% |
| 2028 | 84.4% | 73.9% | 87.0% |
| 2029 | 83.6% | 58.5% | 88.6% |
| 2030 | 82.8% | 41.2% | 90.1% |

2026 y 2027 son iguales en los tres escenarios porque el modelo usa la
cobertura del año anterior: una intervención que arranca en 2026 solo se
refleja en las predicciones a partir de 2028. Esto tiene una implicancia de
política directa: una campaña de recuperación de cobertura iniciada hoy no
muestra efecto estadístico antes de dos años.

## 6. Limitaciones

- El dataset tiene resolución **anual** (n=25 tras crear variables rezagadas), por
  lo que las estimaciones de desempeño tienen alta varianza. Se usó LOOCV para
  mitigar esto, pero los resultados deben leerse como exploratorios.
- Varias series anteriores a 2015 combinan datos oficiales con interpolación
  cuando no había publicaciones digitalizadas; los datos 2015-2025 son los más
  confiables y los que sostienen las conclusiones principales del proyecto.
- El modelo no incorpora variación provincial; Argentina tiene heterogeneidad
  sanitaria significativa entre jurisdicciones que este enfoque nacional no captura.

## 7. Cómo publicar este repositorio (para obtener una URL pública)

**Opción A — GitHub (recomendada):**
```bash
cd proyecto_ml_argentina
git init
git add .
git commit -m "Proyecto final: predicción de riesgo epidemiológico Argentina"
# Crear un repo vacío en https://github.com/new (público) y luego:
git remote add origin https://github.com/<tu-usuario>/<nombre-repo>.git
git branch -M main
git push -u origin main
```
Tu URL pública será `https://github.com/<tu-usuario>/<nombre-repo>`.

**Opción B — Google Colab (sin necesidad de git):**
1. Subí `notebooks/proyecto_final_ml.ipynb` a Google Drive.
2. Abrilo con Google Colab.
3. Usá "Compartir" → "Cualquier persona con el enlace" para obtener una URL pública.

**Opción C — GitHub Gist (solo el notebook, rápido):**
1. Entrá a https://gist.github.com
2. Subí `proyecto_final_ml.ipynb` como gist público.
3. GitHub renderiza automáticamente el notebook en la URL del gist.

## 8. Próximos pasos

1. ~~Desagregar el dataset por provincia~~ → **ver `notebooks/anexo_provincial.ipynb`**.
   Se consiguieron datos reales verificados para 7 jurisdicciones + el promedio
   nacional (Chequeado/RPI, sobre datos del Ministerio de Salud), suficientes para
   un ranking cross-sectional pero NO para el panel completo de 24×26 necesario
   para modelar series de tiempo por provincia. Los portales oficiales
   (datos.salud.gob.ar, Nomivac) bloquean la descarga automatizada vía
   `robots.txt`; la sección 3 del anexo trae el código listo para procesar el
   archivo completo en cuanto se descargue manualmente y se coloque en `data/`.
2. Incorporar series mensuales de cobertura desde Nomivac.
3. Complementar con modelos de series de tiempo (ver `extras/pipeline_vacunacion_reemergentes.py`,
   que usa ARIMA para proyectar casos de coqueluche y sarampión a 2026-2028).

## 9. Anexo provincial — fuentes específicas

| Jurisdicción | Período | Indicador | Fuente |
|---|---|---|---|
| Buenos Aires (prov.), CABA, Santa Fe, Chubut, Jujuy, Corrientes, Argentina (nacional) | 2019 vs. 2023 | Triple viral 2ª dosis (inicio escolar) | Chequeado — Centro de Datos / Red Federal de Periodismo e Innovación, sobre datos del Ministerio de Salud de la Nación |
| Córdoba | 2019 vs. 2024 | Refuerzo triple viral (5 años) — indicador y período distintos, no estrictamente comparable | Chequeado/RPI sobre datos del Ministerio de Salud de Córdoba |

Fuente original: https://chequeado.com/el-explicador/brote-de-sarampion-ninguna-provincia-argentina-llega-al-95-de-cobertura-de-vacuna-triple-viral-como-recomienda-la-ops/

## 10. Conclusión narrativa final

Ver [`notebooks/conclusion_final.ipynb`](notebooks/conclusion_final.ipynb), que
sintetiza en una sola narrativa los tres análisis del proyecto (modelo
nacional, anexo provincial y predicción a 5 años) a la luz de la hipótesis de
trabajo.

Todo este trabajo arrancó con una pregunta simple: ¿la caída de cobertura de
vacunación que Argentina viene sufriendo desde 2022 es solo una preocupación
cualitativa, o es una señal que se puede medir y anticipar? La hipótesis que
se puso a prueba fue que la vacunación funciona como un canario en la mina:
que su deterioro no es solo un síntoma del momento, sino un indicador
adelantado, con un año de anticipación, del riesgo de que reaparezcan
enfermedades que el país creía controladas.

La evidencia nacional respalda esa hipótesis con un grado de solidez que no
es perfecto, pero tampoco es casualidad. Un modelo de clasificación
entrenado solo con la cobertura del año anterior y el contexto
socioeconómico logró distinguir los años de riesgo epidemiológico alto con
un AUC de 0,78, bien por encima del 0,50 que tendría el azar. Y no fue una
variable cualquiera la que explicó esa capacidad: el análisis SHAP señaló a
la cobertura de triple viral rezagada como el factor de mayor peso, por
encima de la inflación, el desempleo y la pobreza. Cuando se reconstruye la
trayectoria histórica de ese riesgo predicho, el patrón es revelador: la
probabilidad ya marcaba 62% en 2023, con la cobertura recién derrumbándose, y
llegó a 80% en 2025, el año en que el sarampión efectivamente volvió a
circular y la tos convulsa tuvo su peor registro en años. El modelo no
descubrió el problema después de que pasara; lo venía anticipando.

Cuando se mira el mapa del país en vez del promedio nacional, la historia se
vuelve más concreta y más urgente. De las jurisdicciones con datos
verificados, Buenos Aires -la provincia más poblada del país- es la que
muestra la caída de cobertura más pronunciada, de 77,3% a 41,9%, varios
puntos por debajo del 46% que el modelo nacional usa como punto de partida.
No es un dato menor: es justamente en Buenos Aires y CABA donde se
confirmaron los primeros casos del brote de 2025. La señal más fuerte y el
desenlace más temprano coincidieron en el mismo lugar, lo que sugiere que el
promedio nacional, aunque estadísticamente válido, diluye un riesgo que está
mucho más concentrado de lo que parece a primera vista.

La proyección a cinco años es, quizás, la parte más útil para decidir qué
hacer con toda esta evidencia. Si nada cambia respecto a 2025 -el escenario
de statu quo-, la probabilidad de un año de riesgo alto se mantiene firme
entre 83% y 86% hasta 2030: no hay alivio espontáneo, el problema no se
resuelve solo. Si la cobertura lograra recuperarse gradualmente hasta
70-72% para 2030, esa probabilidad cae a la mitad, a 41%. Y si la caída
reciente continuara a su ritmo actual, la probabilidad sigue subiendo hasta
90%. Pero hay un detalle que cambia la lectura de cualquier decisión: como
el modelo mira la cobertura del año anterior, una campaña de recuperación
que empezara hoy recién se nota en las cifras a partir de dos años después.
El costo de demorar la decisión no se paga el año que se demora, se paga el
año siguiente.

La conclusión que conecta todo esto con la hipótesis original es que el
canario sí canta, y canta con tiempo de sobra para actuar: la caída de
cobertura de 2023-2025 anticipó estadísticamente el brote que confirmó la
realidad, la geografía del problema señala con bastante claridad dónde
concentrar la respuesta, y la proyección a cinco años muestra que la
diferencia entre un futuro de riesgo sostenido y uno de riesgo reducido a la
mitad depende, en gran parte, de si la recuperación de cobertura empieza
ahora o se posterga otra vez. La pregunta que cerraba la hipótesis -si
alguien está mirando estos datos a tiempo para actuar- ya tiene, con este
proyecto, una respuesta cuantitativa y no solo una corazonada.

**Limitación que vale la pena repetir una última vez:** todo esto se
construyó sobre 25 observaciones anuales a nivel nacional y un recorte
parcial de 7 provincias con apenas 2-3 puntos temporales cada una. Es
evidencia sólida para argumentar y para orientar prioridades, no un sistema
de predicción listo para producción. Su valor está en mostrar que la señal
existe y se puede capturar, no en garantizar el número exacto que va a pasar
en 2028.
