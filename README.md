# 🍄 Mushroom Toxicity Classification

Este repositorio contiene un proyecto de Machine Learning para la predicción y clasificación de la toxicidad en hongos (**Comestible vs. Venenoso**), utilizando datos morfológicos y características ambientales basadas en el conjunto de datos de Wagner (2020) y el repositorio UCI Machine Learning.

---

## 📁 Estructura del Proyecto

La estructura del repositorio sigue el estándar de la industria para proyectos de Ciencia de Datos y Machine Learning:

```text
mushroom-toxicity-classification/
├── data/
│   ├── raw/                            # Datos originales (primary_data, secondary_data y metadatos)
│   └── processed/                      # Datos limpios y estandarizados listos para modelar
├── notebooks/                          # Cuadernos Jupyter ordenados por fases
│   ├── 01_eda.ipynb                    # Análisis Exploratorio de Datos (EDA)
│   ├── 02_preprocessing.ipynb          # Pipeline de limpieza, codificación e imputación
│   └── 03_model_training.ipynb         # Entrenamiento, validación y comparación de modelos
├── src/                                # Código fuente modular y reutilizable
│   ├── __init__.py
│   ├── data_loader.py                  # Carga, validación y división de datasets
│   ├── models.py                       # Definición de pipelines y modelos (Random Forest, etc.)
│   ├── validation.py                   # Validación cruzada estratificada sin fuga de datos
│   ├── especies.py                     # Catálogo de las 173 especies y detector de especímenes fuera de él
│   └── utils.py                        # Métricas, matrices de confusión y gráficos
├── app/                                # Interfaz de usuario interactiva (ESPORA)
│   ├── app.py                          # Aplicación en Streamlit
│   ├── tema.py                         # Identidad visual: paleta, tipografía, estilos y componentes
│   ├── assets/                         # Logos (isotipo, logotipo, favicon)
│   └── models/                         # Metadatos del modelo (los .joblib se generan al entrenar)
├── .streamlit/config.toml              # Tema oscuro de la app
├── docs/                               # Documentación del proyecto
│   └── resultados_eda/                 # Tablas y gráficos generados por la inspección de datos
├── dataset_inspector.py                # Script de auditoría de datos (13 controles de calidad)
├── recalibrar_umbral_cv.py             # Recalibra el umbral τ* con validación cruzada 5-fold
├── validar_por_especie.py              # Mide el rendimiento con especies que el modelo no vio
├── .gitignore                          # Archivos y carpetas ignoradas por Git
├── README.md                           # Documentación principal
└── requirements.txt                    # Dependencias del entorno
```

---

## 📊 Descripción del Conjunto de Datos

El conjunto de datos principal utilizado es `secondary_data.csv` (`data/raw/secondary_data.csv`):
- **Instancias:** 61,069 registros de hongos hipotéticos basados en 173 especies.
- **Variables:** 21 atributos (3 métricos numéricos y 17 nominales categóricos + variable objetivo).
- **Variable Objetivo (`class`):**
  - `p` (*poisonous* / venenoso o no recomendado): ~55.5%
  - `e` (*edible* / comestible): ~44.5%

---

## 🚀 Instalación y Uso

### 1. Clonar el repositorio y crear entorno virtual
```bash
git clone https://github.com/JercOmg/mushroom-toxicity-classification.git
cd mushroom-toxicity-classification

python -m venv .venv
# En Windows:
.venv\Scripts\activate
# En Linux/macOS:
# source .venv/bin/activate
```

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 3. Ejecutar la auditoría e inspección de datos
```bash
python dataset_inspector.py
```

### 4. Entrenar el modelo (obligatorio la primera vez)
Los modelos `.joblib` no se suben al repositorio (`.gitignore`), así que hay que generarlos. El orden importa:
el notebook escribe un umbral provisional y los dos scripts lo corrigen y completan los metadatos.
```bash
cd notebooks
jupyter nbconvert --to notebook --execute 03_model_training.ipynb --output 03_model_training.executed.ipynb
cd ..
python recalibrar_umbral_cv.py
python validar_por_especie.py
```

### 5. Iniciar la aplicación interactiva (Streamlit)
Ejecutar desde la raíz del repositorio para que cargue el tema de `.streamlit/config.toml`:
```bash
streamlit run app/app.py
```

---

## 🔬 Cómo decide la app

El modelo acierta el 100 % con **especies que ya vio**, pero con especies nuevas (validación cruzada
agrupada por especie) baja a ~62 % de accuracy y ROC-AUC 0,64: reconoce especies, no «entiende» la toxicidad.
Por eso el dictamen tiene tres estados:

| Dictamen | Cuándo |
|---|---|
| **Venenoso** | El espécimen encaja 20/20 en una especie conocida y P(venenoso) ≥ franja superior |
| **Comestible** | Encaja 20/20 en una especie conocida y P(venenoso) ≤ franja inferior |
| **Incierto** | No encaja en ninguna especie del catálogo, o P(venenoso) cae en la franja sin datos de entrenamiento |

Con especies desconocidas solo el 0,62 % de los ejemplares pasa el filtro del catálogo, y entre ellos no hubo
venenosos tomados por comestibles. Es un proyecto académico sobre datos simulados: **no sirve para decidir si
un hongo silvestre se puede comer**.