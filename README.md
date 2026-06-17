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


## 2. Estructura del repositorio

```
proyecto_ml_argentina/
├── README.md                          ← este archivo
├── requirements.txt                   ← dependencias con versiones fijas
├── data/
│   └── argentina_dataset.csv          ← dataset histórico 2000-2025 (14 variables)
├── notebooks/
│   ├── proyecto_final_ml.py           ← código fuente (formato jupytext, versionable en git)
│   └── proyecto_final_ml.ipynb        ← notebook ejecutado con todos los outputs
├── img/                                ← gráficos exportados (EDA, ROC, SHAP, etc.)
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

## 6. Limitaciones

- El dataset tiene resolución **anual** (n=25 tras crear variables rezagadas), por
  lo que las estimaciones de desempeño tienen alta varianza. Se usó LOOCV para
  mitigar esto, pero los resultados deben leerse como exploratorios.
- Varias series anteriores a 2015 combinan datos oficiales con interpolación
  cuando no había publicaciones digitalizadas; los datos 2015-2025 son los más
  confiables y los que sostienen las conclusiones principales del proyecto.
- El modelo no incorpora variación provincial; Argentina tiene heterogeneidad
  sanitaria significativa entre jurisdicciones que este enfoque nacional no captura.

