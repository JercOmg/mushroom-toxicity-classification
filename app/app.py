import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="Clasificación de Toxicidad de Hongos",
    page_icon="🍄",
    layout="wide"
)

st.title("🍄 Mushroom Toxicity Classification App")
st.markdown(
    """
    Bienvenido al clasificador inteligente de hongos.
    Esta aplicación permite predecir si un ejemplar de hongo es **Comestible (Edible)** o **Venenoso (Poisonous)** 
    según sus características morfológicas y de hábitat.
    """
)

st.sidebar.header("Parámetros del Hongo")

# Carga de datos base de ejemplo
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "secondary_data.csv"

if DATA_PATH.exists():
    df_sample = pd.read_csv(DATA_PATH, sep=";", nrows=50)
    st.subheader("Vista previa del conjunto de datos")
    st.dataframe(df_sample.head(10))
else:
    st.info("Coloca el dataset en `data/raw/secondary_data.csv` para ver muestras.")

st.info("💡 Puedes entrenar y cargar un modelo desde `notebooks/03_model_training.ipynb` para habilitar predicciones en tiempo real.")
