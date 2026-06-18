# Predicción de riesgo epidemiológico en Argentina a partir de cobertura de vacunación

**Proyecto Final — Data Science II**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/estemartinez87-cell/ENTREGA-FINAL-DATA-SCIENCE-II/blob/main/proyecto_final_ml.ipynb)

## Contexto

Entre 2015 y 2021 Argentina sostuvo coberturas de vacunación infantil por encima del 85%. A partir de 2022 esa cobertura cayó de forma marcada (la segunda dosis de triple viral pasó de ~90% a 46% hacia 2024) y, en simultáneo, reaparecieron brotes de sarampión y un récord de casos de coqueluche. Este proyecto pone a prueba la hipótesis de que **la caída de cobertura de un año funciona como señal de alerta temprana del riesgo epidemiológico del año siguiente**, especialmente cuando coincide con deterioro socioeconómico.

## Objetivo

Clasificación binaria supervisada para anticipar **años de alto riesgo epidemiológico**, con el fin de priorizar campañas de vacunación y recursos de vigilancia epidemiológica antes de que se confirme un brote.

- **Variable objetivo (`riesgo_alto`):** `1` si hubo casos de sarampión confirmados **o** los casos de coqueluche superaron el percentil 75 histórico; `0` en caso contrario.
- **Métrica principal:** AUC-ROC (apropiada para clases desbalanceadas y sin umbral fijo). Métricas secundarias: Accuracy, F1, matriz de confusión.

## Datos

Panel anual de Argentina, 2000–2025 (26 años × 13 columnas), construido a partir de:

| Fuente | Variables |
|---|---|
| INDEC | pobreza, indigencia, desempleo, inflación, tipo de cambio |
| DEIS / Ministerio de Salud | mortalidad infantil, esperanza de vida, fecundidad adolescente |
| SAP / Boletín Epidemiológico Nacional / OPS | casos de coqueluche y sarampión |
| Nomivac / SAP | coberturas de vacunación (triple viral 1ª y 2ª dosis, polio) |

El dataset principal (`argentina_dataset.csv`) se carga directamente desde el repositorio de GitHub. Para el anexo provincial se usa un segundo archivo (`cobertura_provincial_2019_2023.csv`) con datos verificados de 7 jurisdicciones, publicados por el Centro de Datos de Chequeado en base a cifras del Ministerio de Salud.

## Metodología

1. **EDA:** revisión de tipos de datos, estadísticos descriptivos, distribución/outliers y matriz de correlación.
2. **Ingeniería de atributos:**
   - Variables de cobertura **rezagadas un año** (`cobertura_tv2_lag1`, `cobertura_polio_lag1`) para reflejar la causalidad real y evitar fuga de información.
   - Transformación `log1p` para `inflación` y `tipo de cambio` (alta curtosis/asimetría).
   - Tendencia temporal normalizada (`anio_norm`).
   - Variable objetivo compuesta a partir de dos enfermedades centinela (sarampión + coqueluche).
3. **Features finales del modelo:** `cobertura_tv2_lag1`, `cobertura_polio_lag1`, `pobreza`, `desempleo`, `inflacion_log`, `fecundidad_adolescente`, `anio_norm`.
4. **Entrenamiento:** dado el tamaño reducido de la muestra (n≈25), se compararon tres modelos optimizados con **GridSearchCV** (5-fold estratificado, scoring=AUC):
   - Regresión Logística
   - Random Forest
   - XGBoost
5. **Evaluación honesta:** **Leave-One-Out Cross-Validation (LOOCV)**, técnica recomendada para datasets pequeños.

## Resultados

Comparación de modelos (evaluación LOOCV, out-of-fold):

| Modelo | AUC | Accuracy | F1 | Precision | Recall |
|---|---|---|---|---|---|
| **Regresión Logística** | **0.78** | 0.64 | 0.57 | 0.60 | 0.55 |
| Random Forest | 0.68 | 0.56 | 0.42 | 0.50 | 0.36 |
| XGBoost | 0.40 | 0.56 | 0.42 | 0.50 | 0.36 |

**Modelo seleccionado:** Regresión Logística (AUC = 0.78), una capacidad moderada-fuerte para distinguir años de riesgo alto usando solo cobertura rezagada y contexto socioeconómico.

## Explicabilidad (SHAP)

Importancia promedio (|SHAP| medio) de cada variable en el modelo final:

| Variable | Importancia SHAP |
|---|---|
| `cobertura_tv2_lag1` | 0.49 |
| `inflacion_log` | 0.47 |
| `fecundidad_adolescente` | 0.41 |
| `cobertura_polio_lag1` | 0.40 |
| `desempleo` | 0.37 |
| `anio_norm` | 0.33 |
| `pobreza` | 0.15 |

La cobertura de triple viral rezagada es la variable de mayor peso (signo negativo: a menor cobertura el año anterior, mayor probabilidad de alerta al año siguiente), lo que valida la hipótesis central del proyecto.

## Hallazgos principales

- La probabilidad predicha de riesgo alto se mantuvo baja (10–40%) durante 2009–2021, con cobertura siempre por encima de 85%. Tras el colapso de cobertura desde 2022, la probabilidad subió a 62% en 2023 y a 80% en 2025 — año en que efectivamente se confirmó el brote de sarampión y el récord de coqueluche.
- **Anexo provincial (análisis cross-sectional, 7 jurisdicciones):** Buenos Aires muestra la caída de cobertura más pronunciada (de 77.3% a 41.9%), por debajo del promedio nacional. Ninguna jurisdicción con dato verificado alcanza el 95% recomendado por la OPS. Buenos Aires y CABA, además, concentran los primeros casos confirmados del brote 2025. *No se pudo construir el panel completo de 24 jurisdicciones × 26 años porque los portales oficiales (datos.salud.gob.ar, Nomivac) bloquean la descarga automatizada vía `robots.txt`.*
- **Proyección a 5 años (2026–2030):** extendiendo el modelo bajo tres escenarios de cobertura (statu quo, recuperación, deterioro), con variables socioeconómicas congeladas en valores de 2025 y fecundidad adolescente proyectada vía ARIMA:
  - *Statu quo:* riesgo se mantiene en 83–86% hasta 2030.
  - *Recuperación* (cobertura a 70–72% para 2030): riesgo cae a ~41%.
  - *Deterioro continuo:* riesgo sube hasta ~90%.
  - Como el modelo usa cobertura rezagada un año, **una campaña de recuperación iniciada hoy recién se refleja estadísticamente dos años después**.

## Limitaciones

- El dataset nacional tiene resolución anual y solo 26 observaciones (25 tras aplicar el lag): es un ejercicio académico/exploratorio, no un sistema de predicción operativo.
- El anexo provincial cubre solo 7 de 24 jurisdicciones, con 2–3 puntos temporales cada una; no permite modelar series de tiempo ni clasificación por provincia, solo comparación cross-sectional.
- El ARIMA de inflación no convergió de forma confiable, por lo que las variables socioeconómicas se mantienen fijas en la proyección a 5 años (supuesto conservador).

## Estructura del repositorio

```
.
├── proyecto_final_ml.ipynb     # Notebook principal (EDA, modelado, SHAP, anexos)
├── img/                        # Gráficos exportados por el notebook
│   ├── eda_boxplots.png
│   ├── correlation_matrix.png
│   ├── roc_confusion.png
│   ├── shap_summary.png
│   ├── ranking_provincial.png
│   └── prediccion_5_anios.png
└── README.md
```

## Cómo ejecutar

**Opción 1 — Google Colab:** usar el badge "Open in Colab" al inicio de este README (no requiere instalación).

**Opción 2 — Entorno local:**

```bash
pip install pandas numpy matplotlib scipy scikit-learn xgboost shap statsmodels
jupyter notebook proyecto_final_ml.ipynb
```

El notebook descarga los datasets directamente desde este repositorio (`argentina_dataset.csv` y `cobertura_provincial_2019_2023.csv`), por lo que no es necesario descargar archivos por separado.

## Stack tecnológico

Python · pandas · numpy · matplotlib · scipy · scikit-learn · XGBoost · SHAP · statsmodels
