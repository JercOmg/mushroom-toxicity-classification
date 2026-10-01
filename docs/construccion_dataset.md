# Construcción del Dataset

## Sistema Inteligente de Clasificación Micológica — Hito 2

> **Criterio de rúbrica atendido:** *Construcción del Dataset — balanceado acorde al problema y con más de 35.000 registros (40 pts, nivel máximo).*

---

## 1. Fuente de datos

| Propiedad | Valor |
|---|---|
| **Dataset** | Secondary Mushroom Data (`secondary_data.csv`) |
| **Autor** | Dennis Wagner — Fecha: 05 de septiembre de 2020 |
| **Repositorio oficial** | [mushroom.mathematik.uni-marburg.de/files/](https://mushroom.mathematik.uni-marburg.de/files/) |
| **Registro en UCI** | UCI Machine Learning Repository — *Mushroom (Secondary Data)* |
| **Obra de referencia de especies** | Patrick Hardin. *Mushrooms & Toadstools*. Zondervan, 1999 |
| **Licencia / naturaleza** | Datos **hipotéticos simulados** a partir de 173 especies reales (353 especímenes por especie) mediante el módulo `secondary_data_generation.py` del proyecto fuente |

El archivo crudo se encuentra en `data/raw/secondary_data.csv` (separador `;`), acompañado de su ficha de metadatos oficial `data/raw/secondary_data_meta.txt`.

## 2. Volumen y estructura

- **61.069 registros** (instancias de hongos) × **21 columnas**: 20 características + 1 variable objetivo.
- **✅ Cumple el umbral de la rúbrica para el nivel máximo: > 35.000 registros** (supera el umbral en un 74%).

### Variables

| Tipo | Cantidad | Variables |
|---|---|---|
| Métricas (numéricas) | 3 | `cap-diameter` (cm), `stem-height` (cm), `stem-width` (mm) |
| Nominales (categóricas) | 17 | `cap-shape`, `cap-surface`, `cap-color`, `does-bruise-or-bleed`, `gill-attachment`, `gill-spacing`, `gill-color`, `stem-root`, `stem-surface`, `stem-color`, `veil-type`, `veil-color`, `has-ring`, `ring-type`, `spore-print-color`, `habitat`, `season` |
| Objetivo (binaria) | 1 | `class`: `p` = venenoso (poisonous), `e` = comestible (edible) |

## 3. Balance de clases

El dataset es **balanceado acorde al problema**: en toxicología, ambas clases representan poblaciones reales comparables (un hongo hallado en la naturaleza puede ser comestible o tóxico con probabilidad similar), y la clase se distribuye de forma casi equitativa:

| Clase | Registros | Porcentaje |
|---|---|---|
| `p` — Venenoso | 33.888 | 55,49 % |
| `e` — Comestible | 27.181 | 44,51 % |

- **Ratio de desbalance: 1,25 : 1** — dentro del rango considerado *balanceado* en la literatura (por debajo del 1,5 : 1 no se requieren técnicas de sobremuestreo como SMOTE).
- Como salvaguarda adicional, **todas las particiones se realizan con estratificación** (`random_state=42`), lo que preserva exactamente esta proporción en train/validación/test.
- La asimetría residual se gestiona a nivel de decisión con el **umbral de riesgo cero** (Hito 2, notebook 03): el costo del Falso Negativo (venenoso → comestible) es inaceptable y se calibra $\tau^*$ para garantizar $FN = 0$.

> Nota metodológica: el dataset original clasifica especies como *edible*, *poisonous* o *unknown edibility*; la clase desconocida fue **combinada con la venenosa** por los autores, criterio conservador coherente con nuestra aplicación de riesgo cero.

## 4. Auditoría de calidad de datos

La auditoría completa fue automatizada en `dataset_inspector.py` y sus evidencias están en `docs/resultados_eda/` (10 CSV + 7 PNG). Resumen:

### 4.1 Valores nulos

| Columna | Nulos | % |
|---|---|---|
| `veil-type` | 57.892 | 94,8 % |
| `spore-print-color` | 54.715 | 89,6 % |
| `veil-color` | 53.656 | 87,9 % |
| `stem-root` | 51.538 | 84,4 % |
| `stem-surface` | 38.124 | 62,4 % |
| `gill-spacing` | 25.063 | 41,0 % |
| `cap-surface` | 14.120 | 23,1 % |
| `gill-attachment` | 9.884 | 16,2 % |
| `ring-type` | 2.471 | 4,0 % |
| (11 columnas restantes) | 0 | 0 % |

**Decisión:** en el dominio micológico, un valor faltante es **informativo** (la estructura no está presente o no fue observada), no un error de medición. Por eso **no se eliminan filas ni columnas**: se imputan las categóricas con la constante `"missing"`, que One-Hot codifica como categoría propia.

### 4.2 Registros duplicados exactos

- **146 duplicados exactos (0,24 %).** Se **conservan**: el dataset es una simulación de 353 especímenes por especie, donde la coincidencia completa de atributos es esperable; su impacto en 61.069 registros es estadísticamente insignificante y eliminarlos rompería la fidelidad al repositorio fuente.

### 4.3 Outliers en variables numéricas (método IQR)

| Variable | Q1 | Q3 | Límites IQR | Outliers | % |
|---|---|---|---|---|---|
| `cap-diameter` | 3,48 | 8,54 | [−4,11 ; 16,13] | 2.400 | 3,93 % |
| `stem-height` | 4,64 | 7,74 | [−0,01 ; 12,39] | 3.169 | 5,19 % |
| `stem-width` | 5,21 | 16,57 | [−11,83 ; 33,61] | 1.967 | 3,22 % |

**Decisión:** se **conservan** los outliers: corresponden a la variabilidad biológica real (ejemplares gigantes o pequeños) y los modelos basados en árboles (Random Forest, XGBoost) son naturalmente robustos a ellos.

### 4.4 Estadísticas de variables numéricas

| Variable | Media | Desv. Est. | Mín | Máx |
|---|---|---|---|---|
| `cap-diameter` (cm) | 6,73 | 5,26 | 0,38 | 62,34 |
| `stem-height` (cm) | 6,58 | 3,37 | 0,00 | 33,92 |
| `stem-width` (mm) | 12,15 | 10,04 | 0,00 | 103,91 |

## 5. Pipeline ETL aplicado (notebooks 01 y 02)

1. **Extracción (E):** carga de `data/raw/secondary_data.csv` con separador `;` (`src/data_loader.py:load_raw_data`).
2. **Transformación (T):**
   - Codificación del objetivo: $e \mapsto 0$ (comestible), $p \mapsto 1$ (venenoso — clase positiva de riesgo).
   - Imputación: **mediana** para numéricas, **constante `"missing"`** para categóricas.
   - Escalado: `StandardScaler` sobre las 3 numéricas.
   - Codificación: `OneHotEncoder(handle_unknown="ignore")` sobre las 17 categóricas → **128 características** de entrada al modelo.
3. **Carga / partición (L):** división estratificada **70 % entrenamiento / 15 % validación / 15 % prueba** (`src/data_loader.py:get_train_val_test_data`, `random_state=42`), que preserva el balance de clases en cada subconjunto.

El preprocesador se **ajusta exclusivamente con el conjunto de entrenamiento** para evitar fuga de datos (*data leakage*) hacia validación y prueba.

## 6. Trazabilidad de evidencias

| Evidencia | Ubicación |
|---|---|
| Dataset crudo | `data/raw/secondary_data.csv` |
| Ficha oficial del dataset | `data/raw/secondary_data_meta.txt` |
| Auditoría automatizada | `dataset_inspector.py` |
| Reportes de la auditoría (tipos, nulos, duplicados, balance, outliers, categorías) | `docs/resultados_eda/*.csv` |
| Visualizaciones EDA (balance, distribuciones, boxplots) | `docs/resultados_eda/*.png` |
| Exploración analítica | `notebooks/01_eda.ipynb` |
| Preprocesamiento y partición | `notebooks/02_preprocessing.ipynb` |
| Matriz de trazabilidad completa del proyecto | `docs/trazabilidad.md` |
