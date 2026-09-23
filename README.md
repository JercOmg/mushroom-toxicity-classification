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
│   └── utils.py                        # Métricas, matrices de confusión y gráficos
├── app/                                # Interfaz de usuario interactiva
│   └── app.py                          # Aplicación en Streamlit
├── docs/                               # Documentación del proyecto
│   └── resultados_eda/                 # Tablas y gráficos generados por la inspección de datos
├── dataset_inspector.py                # Script de auditoría de datos (13 controles de calidad)
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

### 4. Iniciar la aplicación interactiva (Streamlit)
```bash
streamlit run app/app.py
```