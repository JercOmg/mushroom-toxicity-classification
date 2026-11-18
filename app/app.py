# -*- coding: utf-8 -*-
"""🍄 Sistema Inteligente de Clasificación Micológica — Demo del Hito 2.

Aplicación Streamlit que consume los artefactos entrenados en
`notebooks/03_model_training.ipynb` (app/models/) para clasificar hongos
como comestibles o venenosos con metodología de riesgo cero (FN = 0).

Funcionalidades:
 1.  Predicción individual interactiva (20 características morfológicas).
 2.  Predicción por lotes desde archivo CSV.
 3.  Probabilidad de toxicidad en tiempo real (predict_proba).
 4.  Umbral de riesgo cero τ* configurable con efecto inmediato en el dictamen.
 5.  Desglose local de la probabilidad por variable (contribuciones nativas del Random Forest).
 6.  Simulador "¿qué pasaría si...?" para cualquier variable del espécimen.
 7.  Importancia global precalculada (sin cálculo al arrancar la app).
 8.  Dashboard de métricas del modelo desplegado en el conjunto de test.
 9.  Comparativa de los 4 modelos entrenados.
 10. Matriz de confusión del modelo desplegado.
 11. Exploración del dataset (INFOVIZ: balance, distribuciones, categorías).
 12. Descarga de resultados de predicción por lotes + plantilla CSV.
 13. Ficha técnica del modelo y documentación matemática embebida.
 14. Validación cruzada estratificada k-fold sin fuga de datos (pipeline por fold, test excluido).
 15. Detector de catálogo (src/especies.py) y dictamen en tres estados: comestible, venenoso, incierto.
 16. Rendimiento con especies nuevas (validar_por_especie.py, validación agrupada por especie).

Ejecución:
    streamlit run app/app.py
"""
import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# Sin .resolve(): en Windows la ruta real puede superar los 260 caracteres.
ROOT = Path(__file__).absolute().parent.parent
MODELS_DIR = ROOT / "app" / "models"
DATA_PATH = ROOT / "data" / "raw" / "secondary_data.csv"
FORMULAS_PATH = ROOT / "docs" / "formulas_matematicas_modelos.md"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))
import tema  # noqa: E402
from tema import AMANITA, LIQUEN, LINEA, MUSGO, OCRE, PIZARRA, TIZA, COLORES_MODELOS  # noqa: E402
from src.data_loader import get_train_val_test_data  # noqa: E402
from src.especies import asignar_especie, cargar_catalogo  # noqa: E402
from src.utils import aggregate_by_original, tree_path_contributions, what_if_scan  # noqa: E402
from src.validation import (  # noqa: E402
    METRICAS_CV,
    filas_repetidas_entre,
    resumen_validacion_cruzada,
    validacion_cruzada_modelo,
)

LOGOS = tema.escribir_logos()
st.set_page_config(
    page_title="ESPORA · Toxicidad micológica",
    page_icon=str(LOGOS["favicon"]),
    layout="wide",
)
tema.configurar_matplotlib()
st.markdown(tema.CSS, unsafe_allow_html=True)
SIN_DATO = "— Sin dato"

# ------------------------------------------------------------------ etiquetas legibles
ETIQUETAS_VARIABLES = {
    # Variables métricas (numéricas)
    "cap-diameter": "Diámetro del sombrero (cm)",
    "stem-height": "Altura del tallo (cm)",
    "stem-width": "Ancho del tallo (mm)",
    # Variables nominales (categóricas)
    "cap-shape": "Forma del sombrero",
    "cap-surface": "Superficie del sombrero",
    "cap-color": "Color del sombrero",
    "does-bruise-or-bleed": "¿Presenta magulladuras o sangrado?",
    "gill-attachment": "Inserción de las láminas",
    "gill-spacing": "Espaciado de las láminas",
    "gill-color": "Color de las láminas",
    "stem-root": "Raíz del tallo",
    "stem-surface": "Superficie del tallo",
    "stem-color": "Color del tallo",
    "veil-type": "Tipo de velo",
    "veil-color": "Color del velo",
    "has-ring": "¿Tiene anillo?",
    "ring-type": "Tipo de anillo",
    "spore-print-color": "Color de la esporada",
    "habitat": "Hábitat",
    "season": "Estación / Temporada",
}

ETIQUETAS_NUMERICAS = {
    "cap-diameter": "Diámetro del sombrero (cm)",
    "stem-height": "Altura del tallo (cm)",
    "stem-width": "Ancho del tallo (mm)",
}

ETIQUETAS_CATEGORICAS = {
    "cap-shape": {
        "b": "Campana",
        "c": "Cónica",
        "x": "Convexa",
        "f": "Plana",
        "s": "Hundida",
        "p": "Esférica",
        "o": "Otros",
    },
    "cap-surface": {
        "d": "Seca",
        "e": "Carnosa",
        "g": "Surcos",
        "h": "Brillante",
        "i": "Fibrosa",
        "k": "Sedosa",
        "l": "Coriácea",
        "s": "Lisa",
        "t": "Pegajosa",
        "w": "Arrugada",
        "y": "Escamosa",
    },
    "cap-color": {
        "n": "Marrón",
        "b": "Ante (beige)",
        "g": "Gris",
        "r": "Verde",
        "p": "Rosa",
        "u": "Púrpura",
        "e": "Rojo",
        "w": "Blanco",
        "y": "Amarillo",
        "l": "Azul",
        "o": "Naranja",
        "k": "Negro",
    },
    "does-bruise-or-bleed": {
        "t": "Sí (presenta sangrado o lesión)",
        "f": "No",
    },
    "gill-attachment": {
        "a": "Adnata (adherida)",
        "x": "Adnexa",
        "d": "Decurrente",
        "e": "Libre",
        "s": "Sinuada",
        "p": "Poros",
        "f": "Ninguna",
    },
    "gill-spacing": {
        "c": "Cerradas (apretadas)",
        "d": "Distantes (espaciadas)",
        "f": "Ninguna",
    },
    "gill-color": {
        "n": "Marrón",
        "b": "Ante (beige)",
        "g": "Gris",
        "r": "Verde",
        "p": "Rosa",
        "u": "Púrpura",
        "e": "Rojo",
        "w": "Blanco",
        "y": "Amarillo",
        "l": "Azul",
        "o": "Naranja",
        "k": "Negro",
        "f": "Ninguno",
    },
    "stem-root": {
        "b": "Bulbosa",
        "s": "Hinchada",
        "c": "Claviforme (en maza)",
        "u": "Copa",
        "e": "Igual (cilíndrica)",
        "z": "Rizomorfos",
        "r": "Enraizada",
        "f": "Fibrosa",
    },
    "stem-surface": {
        "d": "Seca",
        "e": "Carnosa",
        "g": "Surcos",
        "h": "Brillante",
        "i": "Fibrosa",
        "k": "Sedosa",
        "l": "Coriácea",
        "s": "Lisa",
        "t": "Pegajosa",
        "w": "Arrugada",
        "y": "Escamosa",
        "f": "Ninguna",
    },
    "stem-color": {
        "n": "Marrón",
        "b": "Ante (beige)",
        "g": "Gris",
        "r": "Verde",
        "p": "Rosa",
        "u": "Púrpura",
        "e": "Rojo",
        "w": "Blanco",
        "y": "Amarillo",
        "l": "Azul",
        "o": "Naranja",
        "k": "Negro",
        "f": "Ninguno",
    },
    "veil-type": {
        "p": "Parcial",
        "u": "Universal",
    },
    "veil-color": {
        "n": "Marrón",
        "b": "Ante (beige)",
        "g": "Gris",
        "r": "Verde",
        "p": "Rosa",
        "u": "Púrpura",
        "e": "Rojo",
        "w": "Blanco",
        "y": "Amarillo",
        "l": "Azul",
        "o": "Naranja",
        "k": "Negro",
        "f": "Ninguno",
    },
    "has-ring": {
        "t": "Sí (con anillo)",
        "f": "No",
    },
    "ring-type": {
        "c": "Araneoso (cortina)",
        "e": "Evanescente (fugaz)",
        "r": "Abocinado (expandible)",
        "g": "Estriado",
        "l": "Grande",
        "p": "Colgante",
        "s": "Envainante",
        "z": "Zonal",
        "y": "Escamoso",
        "m": "Móvil",
        "f": "Ninguno",
    },
    "spore-print-color": {
        "n": "Marrón",
        "b": "Ante (beige)",
        "g": "Gris",
        "r": "Verde",
        "p": "Rosa",
        "u": "Púrpura",
        "e": "Rojo",
        "w": "Blanco",
        "y": "Amarillo",
        "l": "Azul",
        "o": "Naranja",
        "k": "Negro",
    },
    "habitat": {
        "g": "Pastizales",
        "l": "Hojarasca",
        "m": "Praderas",
        "p": "Senderos",
        "h": "Brezales",
        "u": "Urbano",
        "w": "Desechos (vertederos)",
        "d": "Bosques",
    },
    "season": {
        "s": "Primavera",
        "u": "Verano",
        "a": "Otoño",
        "w": "Invierno",
    },
}


def _opcion(columna: str, valor) -> str:
    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
        return SIN_DATO
    mapa = ETIQUETAS_CATEGORICAS.get(columna, {})
    return f"{valor} — {mapa[valor]}" if valor in mapa else str(valor)


def _opciones(columna: str, valores: list) -> list:
    """Formatea los códigos de una columna con su etiqueta legible en español,
    más la opción «sin dato» (muchas especies no tienen, p. ej., raíz o velo)."""
    return [_opcion(columna, v) for v in valores] + [SIN_DATO]


def _codigo(opcion: str):
    return np.nan if opcion == SIN_DATO else opcion.split(" — ")[0].strip()


# ------------------------------------------------------------------ carga de recursos
@st.cache_resource(show_spinner="Cargando modelo y artefactos...")
def cargar_recursos():
    pipeline = joblib.load(MODELS_DIR / "pipeline.joblib")
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")
    model = joblib.load(MODELS_DIR / "best_model.joblib")
    meta = json.loads((MODELS_DIR / "model_metadata.json").read_text(encoding="utf-8"))
    return pipeline, preprocessor, model, meta


@st.cache_data(show_spinner="Cargando dataset...")
def cargar_dataset() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH, sep=";")


@st.cache_resource(show_spinner="Construyendo el catálogo de especies...")
def catalogo_especies(num_cols: tuple, cat_cols: tuple):
    return cargar_catalogo(ROOT, list(num_cols), list(cat_cols))


def dictaminar(filas: pd.DataFrame, probas: np.ndarray) -> pd.DataFrame:
    """Dictamen en tres estados. Solo hay dictamen firme si el espécimen encaja 20/20 en el
    perfil de una especie conocida Y su probabilidad está fuera de la franja sin datos."""
    comp = CATALOGO.compatibilidad(filas[TODAS_LAS_COLS])
    mejor = comp.argmax(axis=1)
    coinciden = comp[np.arange(len(filas)), mejor]
    en_catalogo = coinciden == len(TODAS_LAS_COLS)
    en_franja = (probas > FRANJA[0]) & (probas < FRANJA[1])
    estado = np.where(~en_catalogo | en_franja, "incierto", np.where(probas >= TAU, "venenoso", "comestible"))
    return pd.DataFrame({
        "estado": estado,
        "especie_mas_parecida": [CATALOGO.nombres[i] for i in mejor],
        "rasgos_compatibles": coinciden,
        "en_catalogo": en_catalogo,
        "en_franja": en_franja,
    }, index=filas.index)


def texto_dictamen(estado: str, diag: dict, proba: float) -> str:
    especie = f"<b>{diag['nombre']}</b> <span style='color:{MUSGO}'>({diag['familia']})</span>"
    if not diag["en_catalogo"]:
        return (f"<b>No consumir.</b> El espécimen no encaja en ninguna de las 173 especies conocidas: la más "
                f"parecida es {especie}, con {diag['coinciden']}/{diag['total']} rasgos compatibles. Con especies "
                "que no vio, el modelo acierta apenas 6 de cada 10 veces.")
    if estado == "incierto":
        return (f"<b>No consumir.</b> Encaja con {especie}, pero P(venenoso) = {proba:.0%} cae en la franja "
                f"{FRANJA[0]:.2f}–{FRANJA[1]:.2f}, donde no hubo ningún ejemplar de entrenamiento.")
    if estado == "venenoso":
        return f"<b>No consumir.</b> Perfil compatible 20/20 con {especie}."
    return (f"Perfil compatible 20/20 con {especie}. Es una predicción estadística sobre datos simulados, "
            "<b>no una garantía</b>: ningún hongo silvestre debe consumirse sin la confirmación de un micólogo.")


def _fig_ancho(fig):
    fig.patch.set_facecolor(tema.SUPERFICIE)
    plt.tight_layout()
    return fig


def importancia_global(meta: dict, model, preprocessor, columnas: list) -> pd.Series:
    """Importancia por variable original: la precalculada en el notebook 03 o, si falta,
    las importancias nativas del modelo agrupadas por variable."""
    if meta.get("global_importance"):
        return pd.Series(meta["global_importance"]).sort_values(ascending=False)
    importancias = getattr(model, "feature_importances_", None)
    if importancias is None:
        return None
    agrupadas = aggregate_by_original(importancias, list(preprocessor.get_feature_names_out()), columnas)
    return agrupadas.iloc[0].sort_values(ascending=False)


def barra_contribuciones(contribuciones: pd.Series, titulo: str, top: int = 10):
    """Barras horizontales: rojo empuja hacia venenoso, verde hacia comestible.
    Las variables fuera del top se agrupan en una sola barra."""
    orden = contribuciones.abs().sort_values(ascending=False).index
    visibles = contribuciones[orden[:top]].copy()
    if len(orden) > top:
        visibles[f"otras {len(orden) - top} variables"] = contribuciones[orden[top:]].sum()
    visibles = visibles[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(visibles) + 1.2))
    colores = [AMANITA if v >= 0 else LIQUEN for v in visibles.values]
    barras = ax.barh(visibles.index, visibles.values * 100, color=colores, height=0.62)
    ax.bar_label(barras, labels=[f"{v * 100:+.1f}" for v in visibles.values], padding=4, fontsize=8.5, color=TIZA)
    ax.axvline(0, color=MUSGO, lw=0.8)
    limite = max(abs(visibles.values).max() * 100 * 1.25, 1)
    ax.set_xlim(-limite, limite)
    ax.set_title(titulo)
    ax.set_xlabel("← hacia comestible   ·   puntos porcentuales de P(venenoso)   ·   hacia venenoso →")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, labelcolor=TIZA)
    return _fig_ancho(fig)


def fig_que_pasaria_si(escaneo: pd.DataFrame, columna: str, valor_actual, tau: float, numerica: bool):
    """Probabilidad de toxicidad para cada valor posible de una variable."""
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    nom_col = ETIQUETAS_VARIABLES.get(columna, columna)
    ax.axhspan(FRANJA[0], FRANJA[1], color=OCRE, alpha=0.12, lw=0, label="franja sin datos")
    if numerica:
        ax.plot(escaneo["valor"], escaneo["probabilidad"], color=TIZA, lw=2)
        ax.fill_between(escaneo["valor"], escaneo["probabilidad"], color=TIZA, alpha=0.06)
        ax.axvline(valor_actual, color=LIQUEN, lw=1.2, linestyle=":", label=f"valor actual ({valor_actual:.2f})")
        ax.set_xlabel(nom_col)
    else:
        mapa = ETIQUETAS_CATEGORICAS.get(columna, {})
        etiquetas = [
            "sin\ndato" if pd.isna(v) else f"{v}\n{mapa.get(v, v)}"
            for v in escaneo["valor"]
        ]
        colores = [AMANITA if p >= tau else LIQUEN for p in escaneo["probabilidad"]]
        bordes = [TIZA if (pd.isna(v) and pd.isna(valor_actual)) or v == valor_actual else "none"
                  for v in escaneo["valor"]]
        ax.bar(etiquetas, escaneo["probabilidad"], color=colores, edgecolor=bordes, linewidth=2, width=0.66)
        ax.tick_params(axis="x", labelsize=7.5)
        ax.set_xlabel(f"{nom_col} (contorno claro = valor actual)")
    ax.axhline(tau, color=TIZA, lw=1, linestyle="--", label=f"umbral τ* = {tau:.2f}")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("P(venenoso)")
    ax.set_title(f"¿Qué pasaría si cambia {nom_col}?")
    ax.legend(loc="upper right", ncol=3)
    ax.grid(True, axis="y")
    return _fig_ancho(fig)


def fig_matriz_confusion(cm: dict):
    matriz = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    sns.heatmap(matriz, annot=True, fmt=",d", cmap=tema.mapa_secuencial(LIQUEN), cbar=False,
                linewidths=4, linecolor=tema.SUPERFICIE, annot_kws={"fontsize": 15, "fontweight": "bold"},
                xticklabels=["Comestible", "Venenoso"], yticklabels=["Comestible", "Venenoso"], ax=ax)
    for texto in ax.texts:
        texto.set_color(tema.HUMUS if int(texto.get_text().replace(",", "")) > matriz.max() / 2 else TIZA)
    ax.set_title("Matriz de confusión · test", fontsize=14)
    ax.set_xlabel("Predicción", fontsize=11)
    ax.set_ylabel("Real", fontsize=11)
    ax.tick_params(length=0, labelcolor=TIZA, labelsize=11)
    return _fig_ancho(fig)


def fig_comparativa(df_metricas: pd.DataFrame):
    """Gráfico de puntos: una fila por métrica, un punto por modelo. Más legible que barras
    agrupadas cuando casi todo está entre 0.85 y 1."""
    metricas = ["Accuracy", "Precision", "Recall (Poisonous)", "F1-Score", "ROC-AUC"]
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    for i, metrica in enumerate(metricas):
        valores = df_metricas[metrica].to_numpy()
        ax.plot([valores.min(), valores.max()], [i, i], color=LINEA, lw=10, solid_capstyle="round", zorder=1)
        n = len(df_metricas)
        for k, (_, fila) in enumerate(df_metricas.iterrows()):
            ax.scatter(fila[metrica], i + (k - (n - 1) / 2) * 0.13, s=70,
                       color=COLORES_MODELOS.get(fila["Model"], PIZARRA),
                       edgecolor=tema.SUPERFICIE, linewidth=1.2, zorder=3,
                       label=fila["Model"] if i == 0 else None)
    ax.set_yticks(range(len(metricas)))
    ax.set_yticklabels(metricas, color=TIZA)
    ax.invert_yaxis()
    ax.set_xlim(0.84, 1.008)
    ax.grid(True, axis="x")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Comparación de los 4 modelos · test")
    ax.legend(loc="lower left", ncol=4, bbox_to_anchor=(0, -0.22))
    return _fig_ancho(fig)


# ------------------------------------------------------------------ validación cruzada
SEGUNDOS_POR_FOLD = {"Logistic Regression": 1.7, "Random Forest": 1.6, "XGBoost": 1.4, "MLP Neural Network": 4.2}


@st.cache_data(show_spinner=False)
def particion_desarrollo():
    """Reproduce la partición 70/15/15 del notebook 03 (random_state=42).

    Devuelve el conjunto de desarrollo (Train + Validación), su objetivo binario y
    las comprobaciones de fuga frente al TEST, que nunca entra en la validación cruzada.
    """
    X_tr, X_va, X_te, y_tr, y_va, _ = get_train_val_test_data(cargar_dataset())
    X_dev = pd.concat([X_tr, X_va])
    y_dev = pd.concat([y_tr, y_va]).map({"e": 0, "p": 1}).astype(int)
    controles = {
        "n_desarrollo": len(X_dev),
        "n_test": len(X_te),
        "indices_compartidos": len(X_dev.index.intersection(X_te.index)),
        "test_repetidas_en_desarrollo": filas_repetidas_entre(X_dev, X_te),
        "duplicados_dataset": int(cargar_dataset().duplicated().sum()),
    }
    return X_dev, y_dev, controles


@st.cache_data(show_spinner=False, max_entries=24)
def cv_modelo(nombre_modelo: str, k: int, tau: float) -> pd.DataFrame:
    X_dev, y_dev, _ = particion_desarrollo()
    return validacion_cruzada_modelo(nombre_modelo, X_dev, y_dev, NUM_COLS, CAT_COLS, k=k, tau=tau)


def fig_esquema_folds(k: int):
    fig, ax = plt.subplots(figsize=(8, 0.36 * k + 0.9))
    for fila in range(k):
        y = k - 1 - fila
        for col in range(k):
            valida = fila == col
            ax.add_patch(plt.Rectangle((col, y), 0.94, 0.8, color=LIQUEN if valida else tema.ELEVADO))
            ax.text(col + 0.47, y + 0.4, "valida" if valida else "entrena", ha="center", va="center",
                    fontsize=8 if k <= 5 else 6.5, color=tema.HUMUS if valida else MUSGO)
    ax.set_xlim(-0.05, k)
    ax.set_ylim(-0.1, k)
    ax.set_yticks([k - 1 - f + 0.4 for f in range(k)])
    ax.set_yticklabels([f"Iteración {f + 1}" for f in range(k)], fontsize=8)
    ax.set_xticks([])
    for borde in ax.spines.values():
        borde.set_visible(False)
    ax.set_title(f"{k}-fold estratificado sobre Train + Validación (el test queda fuera)", fontsize=10)
    return _fig_ancho(fig)


def fig_metricas_cv(resultados: pd.DataFrame):
    """Puntos = cada fold; rombo negro = media ± desviación estándar."""
    modelos = list(resultados["Model"].unique())
    metricas = ["Accuracy", "Precision", "Recall (Poisonous)", "F1-Score"]
    fig, ejes = plt.subplots(2, 2, figsize=(10, 1.6 + 0.9 * len(modelos)), sharey=True)
    for ax, metrica in zip(ejes.flat, metricas):
        for i, modelo in enumerate(modelos):
            valores = resultados.loc[resultados["Model"] == modelo, metrica].to_numpy()
            desplazamiento = np.linspace(-0.18, 0.18, len(valores))
            ax.scatter(valores, i + desplazamiento, s=26, alpha=0.8, color=COLORES_MODELOS.get(modelo, PIZARRA))
            ax.errorbar(valores.mean(), i, xerr=valores.std(ddof=1), fmt="D", color=TIZA, ms=5, capsize=4)
        ax.set_yticks(range(len(modelos)))
        ax.set_yticklabels(modelos, fontsize=9, color=TIZA)
        ax.set_title(metrica, fontsize=10)
        ax.grid(True, axis="x")
        ax.tick_params(axis="y", length=0)
        minimo = resultados[metrica].min()
        ax.set_xlim(max(0.0, minimo - 0.02), 1.005)
    ejes[0, 0].invert_yaxis()
    fig.suptitle("Métrica obtenida en cada fold (umbral 0.5)", fontsize=11, fontweight="bold", color=TIZA,
                 x=0.01, ha="left")
    return _fig_ancho(fig)


def fig_tau_por_fold(resultados_modelo: pd.DataFrame, tau: float, nombre_modelo: str):
    taus = resultados_modelo["τ* del fold"].to_numpy()
    etiquetas = [f"Fold {f}" for f in resultados_modelo["Fold"]]
    fig, ax = plt.subplots(figsize=(8, 3.3))
    barras = ax.bar(etiquetas, taus, color=[LIQUEN if t >= tau else AMANITA for t in taus], width=0.55)
    ax.bar_label(barras, labels=[f"{t:.2f}" for t in taus], padding=3, fontsize=9, color=TIZA)
    ax.axhline(tau, color=TIZA, lw=1, linestyle="--", label=f"τ* desplegado = {tau:.2f}")
    ax.set_ylim(0, 1.22)
    ax.set_ylabel("Umbral más alto con FN = 0")
    ax.set_title(f"Umbral más alto sin falsos negativos en cada fold — {nombre_modelo}")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y")
    return _fig_ancho(fig)


def tabla_resumen_cv(resumen: pd.DataFrame) -> pd.DataFrame:
    tabla = pd.DataFrame({"Modelo": resumen["Model"]})
    for metrica in METRICAS_CV:
        tabla[metrica] = [
            f"{m:.4f} ± {d:.4f}" for m, d in zip(resumen[f"{metrica} (media)"], resumen[f"{metrica} (desv.)"])
        ]
    tabla["FN totales (0.5)"] = resumen["FN totales (0.5)"]
    tabla[f"FN totales (τ={TAU:.2f})"] = resumen["FN totales (τ*)"]
    tabla[f"Folds con FN = 0 (τ={TAU:.2f})"] = resumen["Folds con FN = 0 (τ*)"]
    tabla["Umbral más alto con FN = 0 (peor fold)"] = resumen["τ* mínimo entre folds"].map(lambda t: f"{t:.2f}")
    tabla["Tiempo medio por fold (s)"] = resumen["Tiempo medio (s)"].map(lambda t: f"{t:.2f}")
    return tabla


def tabla_cv_vs_test(resumen: pd.DataFrame, metricas_test: list) -> pd.DataFrame:
    """¿El resultado del único split de test cae dentro de la variabilidad de la validación cruzada?"""
    test = pd.DataFrame(metricas_test).set_index("Model")
    filas = []
    for _, r in resumen.iterrows():
        if r["Model"] not in test.index:
            continue
        for metrica in ["Accuracy", "Recall (Poisonous)", "F1-Score", "ROC-AUC"]:
            media, desv = r[f"{metrica} (media)"], r[f"{metrica} (desv.)"]
            valor_test = float(test.loc[r["Model"], metrica])
            filas.append({
                "Modelo": r["Model"],
                "Métrica": metrica,
                "Validación cruzada (media ± desv.)": f"{media:.4f} ± {desv:.4f}",
                "Test (split único)": f"{valor_test:.4f}",
                "¿Test dentro de media ± 2·desv.?": "sí" if abs(valor_test - media) <= 2 * desv + 1e-3 else "no",
            })
    return pd.DataFrame(filas)


# ------------------------------------------------------------------ recursos globales
PIPELINE, PREPROCESSOR, MODEL, META = cargar_recursos()
TAU = float(META["optimal_threshold"])
CALIBRACION = META.get("threshold_calibration") or {}
ORIGEN_TAU = f"validación cruzada {CALIBRACION['k_folds']}-fold" if CALIBRACION else "un split de validación"
if CALIBRACION.get("cv_out_of_fold_band") and "margin_to_edible" in CALIBRACION:
    _franja = CALIBRACION["cv_out_of_fold_band"]
    REGLA_TAU = (
        f"τ* está en el centro de la franja sin errores de la {ORIGEN_TAU} "
        f"({_franja['max_proba_edible']:.2f} – {_franja['min_proba_poisonous']:.2f}): FN = 0 y FP = 0 en "
        f"todos los folds, con un margen de {CALIBRACION['margin_to_edible']:.2f} hacia los comestibles y de "
        f"{CALIBRACION['margin_to_poisonous']:.2f} hacia los venenosos."
    )
else:
    REGLA_TAU = f"τ* es el umbral más alto con 0 falsos negativos en {ORIGEN_TAU}."
NUM_COLS = META["numeric_features"]
CAT_COLS = META["categorical_features"]
TODAS_LAS_COLS = NUM_COLS + CAT_COLS

if not DATA_PATH.exists():
    st.error("No se encontró data/raw/secondary_data.csv. Clona el repositorio completo.")
    st.stop()
DF = cargar_dataset()
FEATURE_NAMES = list(PREPROCESSOR.get_feature_names_out())
MODELO_DE_ARBOLES = hasattr(MODEL, "estimators_") and hasattr(MODEL.estimators_[0], "tree_")
CATALOGO = catalogo_especies(tuple(NUM_COLS), tuple(CAT_COLS))
ESPECIE_FILA = asignar_especie(len(DF))
_banda = CALIBRACION.get("cv_out_of_fold_band") or {}
FRANJA = (float(_banda.get("max_proba_edible", TAU)), float(_banda.get("min_proba_poisonous", TAU)))
VAL_ESPECIE = META.get("species_validation") or {}

# ------------------------------------------------------------------ barra lateral
PAGINAS = {
    "01  Diagnóstico": "Diagnóstico de un espécimen",
    "02  Lotes": "Diagnóstico por lotes",
    "03  Rendimiento": "Rendimiento del modelo",
    "04  Validación": "Validación cruzada",
    "05  Explicabilidad": "¿Por qué decidió esto?",
    "06  Datos": "Exploración del dataset",
    "07  Ficha técnica": "Ficha técnica y fórmulas",
}
with st.sidebar:
    st.markdown(tema.marca_lateral(), unsafe_allow_html=True)
    PAGINA = st.radio("Navegación", list(PAGINAS), label_visibility="collapsed")
    st.markdown(
        f"""<div class="esp-ficha">
MODELO <span>{META["model_name"]}</span><br>
UMBRAL τ* <span>{TAU:.2f}</span><br>
FRANJA INCIERTA <span>{FRANJA[0]:.2f}–{FRANJA[1]:.2f}</span><br>
ESPECIES <span>{len(CATALOGO.nombres)}</span><br>
ENTRENADO <span>{str(META.get("trained_at", "—"))[:10]}</span></div>""",
        unsafe_allow_html=True,
    )
    st.caption(
        "Proyecto académico sobre datos simulados. No usar para decidir si un hongo silvestre se puede comer."
    )

NUMERO, NOMBRE = PAGINA.split("  ")
KICKER = f"ESPORA <b>/</b> {NOMBRE.upper()} <b>·</b> {META['model_name'].upper()} <b>·</b> <span style='text-transform:none'>τ*</span> {TAU:.2f}"
BAJADAS = {
    "01": "Describe el hongo con sus 20 rasgos. El sistema solo da un dictamen si encaja en una de las 173 "
          "especies conocidas; si no, responde «incierto».",
    "02": "Sube un CSV con varios especímenes y descarga el dictamen de cada uno.",
    "03": "Cómo rinde el modelo con especies que ya conoce y con especies que nunca vio.",
    "04": "Validación cruzada estratificada k-fold, sin fuga de datos entre entrenamiento y validación.",
    "05": "Qué rasgos empujaron la probabilidad hacia venenoso o comestible, y qué pasaría si cambian.",
    "06": "61.069 ejemplares simulados de 173 especies — Secondary Mushroom Dataset (Wagner et al., 2021).",
    "07": "Calibración del umbral, metadatos del modelo y documentación matemática.",
}
st.markdown(tema.cabecera(NUMERO, PAGINAS[PAGINA], BAJADAS[NUMERO], KICKER), unsafe_allow_html=True)


def _rellenar_desde_especie():
    """Carga en el formulario un ejemplar real de la especie elegida."""
    nombre = st.session_state.get("plantilla_especie")
    if not nombre or nombre not in CATALOGO.nombres:
        return
    i = CATALOGO.nombres.index(nombre)
    ejemplar = DF.iloc[i * 353 + int(st.session_state.get("plantilla_semilla", 0)) % 353]
    for col in NUM_COLS:
        rango = META["numeric_ranges"][col]
        st.session_state[f"in_{col}"] = float(np.clip(ejemplar[col], rango["min"], rango["max"]))
    for col in CAT_COLS:
        st.session_state[f"in_{col}"] = _opcion(col, ejemplar[col])


# ================================================================== 1. DIAGNÓSTICO
if NUMERO == "01":
    if "in_cap-diameter" not in st.session_state:
        st.session_state["plantilla_especie"] = CATALOGO.nombres[0]
        _rellenar_desde_especie()

    c_pl, c_btn = st.columns([3, 1], vertical_alignment="bottom")
    c_pl.selectbox(
        "Partir de un ejemplar del catálogo",
        CATALOGO.nombres,
        key="plantilla_especie",
        format_func=lambda n: f"{n} · {CATALOGO.familias[CATALOGO.nombres.index(n)]}",
        on_change=_rellenar_desde_especie,
        help="Rellena el formulario con un ejemplar real de esa especie. Después puedes modificar cualquier rasgo.",
    )
    if c_btn.button("Otro ejemplar", width="stretch"):
        st.session_state["plantilla_semilla"] = int(np.random.default_rng().integers(0, 353))
        _rellenar_desde_especie()
        st.rerun()

    with st.form("formulario_hongo"):
        st.markdown(tema.seccion("Medidas"), unsafe_allow_html=True)
        c_num = st.columns(3)
        valores_num = {}
        for col, contenedor in zip(NUM_COLS, c_num):
            rango = META["numeric_ranges"][col]
            valores_num[col] = contenedor.slider(
                ETIQUETAS_NUMERICAS[col],
                min_value=float(rango["min"]),
                max_value=float(rango["max"]),
                step=0.1,
                key=f"in_{col}",
            )

        st.markdown(tema.seccion("Rasgos"), unsafe_allow_html=True)
        valores_cat = {}
        for i in range(0, len(CAT_COLS), 3):
            fila = st.columns(3)
            for col, contenedor in zip(CAT_COLS[i:i + 3], fila):
                valores_cat[col] = contenedor.selectbox(
                    ETIQUETAS_VARIABLES.get(col, col),
                    _opciones(col, META["categorical_values"][col]),
                    key=f"in_{col}",
                )
        ejecutar = st.form_submit_button("Analizar espécimen", width="stretch")

    if ejecutar:
        fila = {**valores_num, **{c: _codigo(v) for c, v in valores_cat.items()}}
        st.session_state["ultima_prediccion"] = {"fila": fila}

    if "ultima_prediccion" in st.session_state:
        fila = st.session_state["ultima_prediccion"]["fila"]
        entrada = pd.DataFrame([fila])[TODAS_LAS_COLS]
        proba = float(PIPELINE.predict_proba(entrada)[0, 1])
        st.session_state["ultima_prediccion"]["proba"] = proba
        estado = dictaminar(entrada, np.array([proba]))["estado"].iloc[0]
        diag = CATALOGO.diagnostico(fila)

        st.markdown(tema.veredicto(estado, texto_dictamen(estado, diag, proba), proba), unsafe_allow_html=True)
        st.markdown(tema.espectro(proba, TAU, FRANJA), unsafe_allow_html=True)

        st.markdown(tema.seccion(
            f"Especie más parecida · {diag['nombre']} · {diag['coinciden']}/{diag['total']} rasgos compatibles"
        ), unsafe_allow_html=True)
        st.markdown(tema.rasgos({c: ETIQUETAS_VARIABLES.get(c, c) for c in TODAS_LAS_COLS},
                                diag["variables_fuera"]), unsafe_allow_html=True)
        if diag["variables_fuera"]:
            st.caption("Los rasgos con contorno ocre no encajan en el perfil observado de esa especie.")

        with st.expander("Explorar otro umbral"):
            tau_usuario = st.slider(
                "Umbral τ", min_value=0.01, max_value=0.95, value=TAU, step=0.01,
                help=f"τ* = {TAU:.2f}. {REGLA_TAU}",
            )
            st.markdown(
                f"Con τ = {tau_usuario:.2f}, el modelo marcaría este espécimen como "
                f"**{'venenoso' if proba >= tau_usuario else 'comestible'}**. El detector de catálogo y la "
                "franja incierta se aplican igual con cualquier umbral."
            )

        with st.expander("Ver características ingresadas"):
            datos_mostrar = []
            for col in TODAS_LAS_COLS:
                val = fila.get(col)
                nom_var = ETIQUETAS_VARIABLES.get(col, col)
                if col in NUM_COLS:
                    unidad = "cm" if "diameter" in col or "height" in col else "mm"
                    val_str = f"{float(val):.2f} {unidad}"
                elif pd.isna(val):
                    val_str = "sin dato"
                else:
                    desc = ETIQUETAS_CATEGORICAS.get(col, {}).get(val, val)
                    if desc != val:
                        val_str = f"{desc} ({val})"
                    else:
                        val_str = str(val)
                datos_mostrar.append({"Característica": nom_var, "Valor": val_str})

            df_caracteristicas = pd.DataFrame(datos_mostrar)
            st.dataframe(df_caracteristicas, width="stretch", hide_index=True)

# ================================================================== 2. LOTES
if NUMERO == "02":
    st.caption("El CSV debe tener las 20 características (la columna `class` es opcional y se ignora).")

    plantilla = DF[TODAS_LAS_COLS].iloc[::12000].copy()
    st.download_button(
        f"Descargar plantilla de ejemplo ({len(plantilla)} registros)",
        plantilla.to_csv(index=False).encode("utf-8"),
        file_name="plantilla_hongos.csv",
        mime="text/csv",
    )

    archivo = st.file_uploader("Archivo CSV", type=["csv"])
    if archivo is not None:
        try:
            lote = pd.read_csv(archivo, sep=None, engine="python")
        except Exception:
            archivo.seek(0)
            lote = pd.read_csv(archivo, sep=";")

        faltantes = [c for c in TODAS_LAS_COLS if c not in lote.columns]
        if faltantes:
            st.error(f"Faltan columnas obligatorias: {', '.join(faltantes)}")
        else:
            with st.spinner("Infiriendo sobre el lote..."):
                probabilidades = PIPELINE.predict_proba(lote[TODAS_LAS_COLS])[:, 1]
            resultado = lote.copy()
            resultado["probabilidad_venenoso"] = probabilidades.round(4)
            dictamen = dictaminar(lote, probabilidades)
            resultado["dictamen"] = dictamen["estado"].map(tema.TITULOS_ESTADO)
            resultado["especie_mas_parecida"] = dictamen["especie_mas_parecida"]
            resultado["rasgos_compatibles"] = dictamen["rasgos_compatibles"].astype(str) + f"/{len(TODAS_LAS_COLS)}"

            conteo = dictamen["estado"].value_counts()
            st.markdown(tema.cifras([
                ("Procesados", f"{len(resultado):,}", "registros del archivo", ""),
                ("Venenosos", f"{conteo.get('venenoso', 0):,}", "no consumir", "mala"),
                ("Comestibles", f"{conteo.get('comestible', 0):,}", "perfil 20/20 de una especie comestible", "buena"),
                ("Inciertos", f"{conteo.get('incierto', 0):,}", "fuera del catálogo o en la franja", ""),
            ]), unsafe_allow_html=True)

            st.dataframe(resultado.head(100), width="stretch")
            if len(resultado) > 100:
                st.caption(f"Mostrando 100 de {len(resultado)} registros.")

            st.download_button(
                "Descargar resultados completos (CSV)",
                resultado.to_csv(index=False).encode("utf-8"),
                file_name="resultados_prediccion.csv",
                mime="text/csv",
            )

# ================================================================== 3. RENDIMIENTO
elif NUMERO == "03":
    mejor = META["best_model_test_metrics"]
    cm_test = META["best_model_confusion_matrix_test"]
    n_test = sum(cm_test.values())
    nuevas = VAL_ESPECIE.get("unseen_species", {})
    detector = VAL_ESPECIE.get("catalog_detector", {})

    st.markdown(tema.seccion("Dos preguntas distintas"), unsafe_allow_html=True)
    st.markdown(tema.cifras([
        ("Especies conocidas · test", f"{mejor['Precision']:.1%}",
         f"precisión sobre {n_test:,} ejemplares de especies vistas al entrenar", "buena"),
        ("Especies nuevas · accuracy", f"{nuevas.get('accuracy', float('nan')):.1%}" if nuevas else "—",
         "validación cruzada agrupada por especie", "mala"),
        ("Especies nuevas · ROC-AUC", f"{nuevas.get('roc_auc', float('nan')):.2f}" if nuevas else "—",
         "0,50 sería azar", "mala"),
        ("Venenosas no detectadas", f"{nuevas.get('species_poisonous_missed', 0)}/"
         f"{nuevas.get('species_poisonous_total', 0)}" if nuevas else "—",
         f"especies nuevas con τ* = {TAU:.2f}", "mala"),
    ]), unsafe_allow_html=True)
    if nuevas:
        st.markdown(tema.nota(
            "<b>Por qué tanta diferencia.</b> El dataset tiene 353 ejemplares simulados por especie, y la "
            "partición aleatoria pone ejemplares de la misma especie en entrenamiento y en prueba. El 100 % "
            "mide si el modelo <b>reconoce especies que ya vio</b>. Cuando cada especie queda entera de un "
            "solo lado (validación cruzada agrupada por especie, k = 5), el rendimiento cae cerca del azar. "
            f"<br><br><b>Cómo lo maneja la app.</b> Solo da un dictamen si el espécimen encaja 20/20 en una "
            f"especie conocida. Con especies desconocidas, solo el {detector.get('unseen_rows_passing', 0):.2%} "
            f"de los ejemplares pasa ese filtro, y entre ellos hubo "
            f"{detector.get('errors_on_passing_rows', {}).get('fn', '—')} venenosos tomados por comestibles. "
            "Todo lo demás se reporta como <b>incierto</b>."
        ), unsafe_allow_html=True)
    else:
        st.warning("Falta la validación por especie: ejecuta `python validar_por_especie.py`.")

    st.markdown(tema.seccion("Especies conocidas · conjunto de prueba"), unsafe_allow_html=True)
    c_matriz, c_cifras = st.columns([1, 1], gap="medium")
    with c_matriz:
        st.pyplot(fig_matriz_confusion(cm_test))
    with c_cifras:
        st.markdown(tema.cifras([
            ("Accuracy", f"{mejor['Accuracy']:.1%}", "aciertos sobre el total", ""),
            ("Precision", f"{mejor['Precision']:.1%}", "de los marcados venenosos, lo eran", ""),
            ("Recall venenoso", f"{mejor['Recall (Poisonous)']:.1%}", "venenosos detectados", ""),
            ("F1-Score", f"{mejor['F1-Score']:.3f}", "balance precision / recall", ""),
            ("ROC-AUC", f"{mejor['ROC-AUC']:.3f}", "capacidad de ordenar por riesgo", ""),
            ("Ejemplares", f"{n_test:,}", "15 % del dataset, partición aleatoria", ""),
        ], columnas=2), unsafe_allow_html=True)
    st.caption(
        f"Selección por evidencia — criterio: {META['selection_criteria']}. "
        f"Umbral de riesgo cero calibrado: τ* = {TAU:.2f}."
    )

    df_test = pd.DataFrame(META["test_metrics"])
    st.markdown(tema.seccion("Los 4 modelos entrenados"), unsafe_allow_html=True)
    st.pyplot(fig_comparativa(df_test))

    st.markdown(tema.seccion("Tabla comparativa · test"), unsafe_allow_html=True)
    st.dataframe(
        df_test.style.format(
            {c: "{:.4f}" for c in ["Accuracy", "Precision", "Recall (Poisonous)", "F1-Score", "ROC-AUC"]}
        ),
        width="stretch",
        hide_index=True,
    )
    with st.expander("Ver métricas de validación (usadas para decisiones de diseño)"):
        st.dataframe(pd.DataFrame(META["validation_metrics"]), width="stretch", hide_index=True)

# ================================================================== 3b. VALIDACIÓN CRUZADA
elif NUMERO == "04":
    st.markdown(
        "Un solo split train/test puede salir «con suerte». La validación cruzada divide el conjunto de "
        "desarrollo en **k partes**, entrena **k veces** dejando fuera una parte distinta cada vez y "
        "reporta la **media ± desviación estándar**. Si la desviación es pequeña, el rendimiento es estable "
        "y no depende de cómo cayó la partición."
    )

    MODELOS_DISPONIBLES = list(COLORES_MODELOS)
    with st.form("formulario_cv"):
        c_mod, c_k = st.columns([2, 1])
        modelos_elegidos = c_mod.multiselect(
            "Modelos a validar", MODELOS_DISPONIBLES, default=MODELOS_DISPONIBLES,
            help="Se entrena un pipeline nuevo (preprocesamiento + modelo) en cada fold.",
        )
        k_elegido = c_k.select_slider("Número de folds (k)", options=[3, 5, 10], value=5)
        estimado = sum(SEGUNDOS_POR_FOLD.get(m, 2) for m in modelos_elegidos) * k_elegido
        st.caption(f"Tiempo estimado la primera vez: ~{estimado:.0f} s. Después queda en caché y es instantáneo.")
        ejecutar_cv = st.form_submit_button("Ejecutar validación cruzada", width="stretch")

    if ejecutar_cv:
        if not modelos_elegidos:
            st.warning("Elige al menos un modelo.")
        else:
            st.session_state["cv_config"] = (tuple(modelos_elegidos), int(k_elegido))

    st.pyplot(fig_esquema_folds(int(k_elegido)))

    config = st.session_state.get("cv_config")
    if config is None:
        st.info("Elige los modelos y el número de folds, y pulsa **Ejecutar validación cruzada**.")
    else:
        modelos_cv, k_cv = config
        X_dev, y_dev, controles = particion_desarrollo()
        partes = []
        with st.status(f"Validación cruzada {k_cv}-fold en curso...", expanded=False) as estado:
            for i, modelo in enumerate(modelos_cv, start=1):
                estado.update(label=f"[{i}/{len(modelos_cv)}] Validando {modelo} ({k_cv} folds)...")
                partes.append(cv_modelo(modelo, k_cv, TAU))
            estado.update(label=f"Validación cruzada {k_cv}-fold completada", state="complete")
        resultados_cv = pd.concat(partes, ignore_index=True)
        resumen_cv = resumen_validacion_cruzada(resultados_cv)

        st.markdown(tema.seccion(f"Resultados: {k_cv}-fold sobre {controles['n_desarrollo']:,} especímenes de desarrollo"), unsafe_allow_html=True)
        st.dataframe(tabla_resumen_cv(resumen_cv), width="stretch", hide_index=True)
        st.caption(
            "Métricas con el umbral 0.5. Las columnas con τ aplican el umbral de riesgo cero "
            f"τ* = {TAU:.2f}, calibrado para el modelo desplegado ({META['model_name']}); en los demás "
            "modelos es solo referencia. «Umbral más alto con FN = 0 (peor fold)» es el umbral más alto "
            "que habría dado FN = 0 en todos los folds: el borde superior de la franja segura."
        )

        # ---------------------------------------------------- dictamen sobre el modelo desplegado
        desplegado = META["model_name"]
        if desplegado in modelos_cv:
            res_desp = resultados_cv[resultados_cv["Model"] == desplegado]
            fn_tau = int(res_desp["FN (τ*)"].sum())
            folds_con_fn = int((res_desp["FN (τ*)"] > 0).sum())
            tau_robusto = float(res_desp["τ* del fold"].min())
            if fn_tau == 0:
                st.success(
                    f"Con τ* = {TAU:.2f}, el {desplegado} no dejó pasar ningún hongo venenoso en ninguno de "
                    f"los {k_cv} folds (con especies que ya conocía; ver la nota sobre especies nuevas abajo)."
                )
            else:
                st.warning(
                    f"Con τ* = {TAU:.2f}, el {desplegado} clasificó como comestibles **{fn_tau}** "
                    f"hongo(s) venenoso(s) en **{folds_con_fn} de {k_cv}** folds. El τ* desplegado se calibró "
                    f"con {ORIGEN_TAU}; con esta configuración, el umbral más alto con FN = 0 en todos los "
                    f"folds es **{tau_robusto:.2f}**. Para recalibrarlo, ejecuta `python recalibrar_umbral_cv.py`."
                )
            st.pyplot(fig_tau_por_fold(res_desp, TAU, desplegado))
            st.caption(
                "Cada barra es el umbral más alto que no deja pasar venenosos en ese fold. Verde: τ* desplegado "
                "queda por debajo, así que no produce falsos negativos. Rojo: sí los produce."
            )

        st.markdown(tema.seccion("Variabilidad entre folds"), unsafe_allow_html=True)
        st.pyplot(fig_metricas_cv(resultados_cv))

        st.markdown(tema.seccion("¿El resultado del test fue suerte?"), unsafe_allow_html=True)
        st.caption(
            "Compara la métrica del único split de test (notebook 03) con la media ± desviación de la "
            "validación cruzada. Si el test cae dentro de la banda, el resultado reportado es representativo."
        )
        st.dataframe(tabla_cv_vs_test(resumen_cv, META["test_metrics"]), width="stretch", hide_index=True)

        with st.expander("Ver resultados fold por fold"):
            st.dataframe(
                resultados_cv.style.format(
                    {**{m: "{:.4f}" for m in METRICAS_CV}, "% venenosos (validación)": "{:.2%}",
                     "τ* del fold": "{:.2f}", "Tiempo (s)": "{:.2f}"}
                ),
                width="stretch",
                hide_index=True,
            )
            st.download_button(
                "Descargar resultados por fold (CSV)",
                resultados_cv.to_csv(index=False).encode("utf-8"),
                file_name=f"validacion_cruzada_{k_cv}fold.csv",
                mime="text/csv",
            )

        # ---------------------------------------------------- controles de fuga de datos
        st.markdown(tema.seccion("Controles de fuga de datos (data leakage)"), unsafe_allow_html=True)
        prop = resultados_cv["% venenosos (validación)"]
        repetidas = controles["test_repetidas_en_desarrollo"]
        st.markdown(
            f"""
| Control | Resultado |
|---|---|
| El test no participa en la validación cruzada | Sí. {controles['n_desarrollo']:,} filas de desarrollo y {controles['n_test']:,} de test; índices compartidos: **{controles['indices_compartidos']}** |
| Preprocesamiento ajustado dentro de cada fold | Sí. Se crea un `Pipeline` nuevo por fold: el imputador, el escalador y el One-Hot solo ven las filas de entrenamiento del fold |
| Folds estratificados | Sí. % de venenosos en validación entre **{prop.min():.2%}** y **{prop.max():.2%}** |
| Filas del test idénticas a filas de desarrollo | {'Ninguna' if repetidas == 0 else f'**{repetidas}** ({repetidas / controles["n_test"]:.2%} del test) — el dataset tiene {controles["duplicados_dataset"]} registros duplicados'} |
"""
        )
        st.code(
            "for idx_ent, idx_val in StratifiedKFold(n_splits=k, shuffle=True, random_state=42).split(X_dev, y_dev):\n"
            "    pipeline = create_model_pipeline(clone(modelo), columnas_numericas, columnas_categoricas)\n"
            "    pipeline.fit(X_dev.iloc[idx_ent], y_dev.iloc[idx_ent])      # se ajusta solo con el fold de entrenamiento\n"
            "    proba = pipeline.predict_proba(X_dev.iloc[idx_val])[:, 1]   # se evalúa con datos que no vio",
            language="python",
        )
        nuevas = VAL_ESPECIE.get("unseen_species", {})
        st.markdown(tema.nota(
            "<b>Limitación medida, no supuesta.</b> Esta validación cruzada reparte al azar ejemplares de la misma "
            "especie entre entrenamiento y validación, así que mide si el modelo reconoce especies que ya vio. "
            + (f"Agrupando por especie (ninguna especie a ambos lados), el {META['model_name']} baja a "
               f"<b>{nuevas['accuracy']:.1%}</b> de accuracy y ROC-AUC <b>{nuevas['roc_auc']:.2f}</b>. Detalle en "
               "«03 Rendimiento»." if nuevas else "Ejecuta `python validar_por_especie.py` para medirlo.")
        ), unsafe_allow_html=True)

# ================================================================== 4. EXPLICABILIDAD
elif NUMERO == "05":
    ultima = st.session_state.get("ultima_prediccion")
    if ultima is None:
        espécimen = DF[TODAS_LAS_COLS].iloc[0].to_dict()
        st.caption("No hay un análisis reciente: se explica el primer ejemplar del dataset.")
    else:
        espécimen = ultima["fila"]
        st.caption("Explicando tu último análisis de «01 Diagnóstico».")

    entrada = pd.DataFrame([espécimen])[TODAS_LAS_COLS]
    proba = float(PIPELINE.predict_proba(entrada)[0, 1])
    venenoso = proba >= TAU
    estado = dictaminar(entrada, np.array([proba]))["estado"].iloc[0]

    st.markdown(tema.cifras([
        ("Dictamen", f'<span style="color:{tema.COLOR_ESTADO[estado]}">{tema.TITULOS_ESTADO[estado]}</span>',
         "con el detector de catálogo", ""),
        ("P(venenoso)", f"{proba:.1%}", "probabilidad del modelo", ""),
        ("Umbral τ*", f"{TAU:.2f}", f"franja incierta {FRANJA[0]:.2f}–{FRANJA[1]:.2f}", ""),
    ], columnas=3), unsafe_allow_html=True)
    if estado == "incierto":
        st.caption("El dictamen es «incierto»: la explicación muestra cómo razona el modelo, pero su "
                   "probabilidad no es fiable para este espécimen.")

    # ---------------------------------------------------------- desglose local
    st.markdown(tema.seccion("1. Desglose de la probabilidad por variable"), unsafe_allow_html=True)
    if MODELO_DE_ARBOLES:
        base, contrib = tree_path_contributions(MODEL, PREPROCESSOR.transform(entrada))
        por_variable = aggregate_by_original(contrib, FEATURE_NAMES, TODAS_LAS_COLS).iloc[0]
        st.caption(
            f"Punto de partida **{base:.1%}** (proporción de venenosos con la que arrancan los árboles) "
            f"+ aporte de cada variable = **{base + por_variable.sum():.1%}**. "
            "Rojo empuja hacia venenoso; verde, hacia comestible."
        )
        por_variable_es = por_variable.copy()
        por_variable_es.index = [ETIQUETAS_VARIABLES.get(c, c) for c in por_variable_es.index]
        st.pyplot(barra_contribuciones(por_variable_es, "Aporte de cada variable a P(venenoso)"))
        with st.expander("Ver el detalle por característica codificada (One-Hot)"):
            detalle = pd.DataFrame({"característica": FEATURE_NAMES, "aporte": contrib[0]})
            detalle = detalle[detalle["aporte"] != 0].sort_values("aporte", key=np.abs, ascending=False)
            st.dataframe(detalle.style.format({"aporte": "{:+.4f}"}), width="stretch", hide_index=True)
    else:
        st.info(
            f"El desglose nativo solo aplica a ensambles de árboles; el modelo desplegado es "
            f"{META['model_name']}. Usa el simulador de abajo para analizar la decisión."
        )

    # ---------------------------------------------------------- simulador qué pasaría si
    st.markdown(tema.seccion("2. ¿Qué pasaría si…?"), unsafe_allow_html=True)
    st.caption("Cambia una variable del espécimen a todos sus valores posibles y observa la probabilidad.")
    columna = st.selectbox(
        "Variable a simular",
        TODAS_LAS_COLS,
        format_func=lambda c: ETIQUETAS_VARIABLES.get(c, c),
        index=TODAS_LAS_COLS.index(por_variable.abs().idxmax()) if MODELO_DE_ARBOLES else 0,
    )
    numerica = columna in NUM_COLS
    if numerica:
        rango = META["numeric_ranges"][columna]
        valores = np.linspace(rango["min"], rango["max"], 60).tolist()
    else:
        valores = META["categorical_values"][columna] + [np.nan]
    escaneo = what_if_scan(PIPELINE, espécimen, columna, valores)
    st.pyplot(fig_que_pasaria_si(escaneo, columna, espécimen[columna], TAU, numerica))
    cambian = escaneo[(escaneo["probabilidad"] >= TAU) != venenoso]
    if cambian.empty:
        st.success(f"Ningún valor de `{columna}` cambia el dictamen por sí solo.")
    else:
        st.warning(
            f"{len(cambian)} de {len(escaneo)} valores de `{columna}` cambiarían el dictamen a "
            f"**{'comestible' if venenoso else 'venenoso'}**."
        )

    # ---------------------------------------------------------- importancia global
    st.markdown(tema.seccion("3. Importancia global del modelo"), unsafe_allow_html=True)
    importancia = importancia_global(META, MODEL, PREPROCESSOR, TODAS_LAS_COLS)
    if importancia is None:
        st.warning("Este modelo no expone importancias globales.")
    else:
        top = importancia.head(12)[::-1].copy()
        top.index = [ETIQUETAS_VARIABLES.get(c, c) for c in top.index]
        fig, ax = plt.subplots(figsize=(8, 4.6))
        ax.hlines(top.index, 0, top.values * 100, color=LINEA, lw=2)
        ax.scatter(top.values * 100, top.index, s=70, color=LIQUEN, zorder=3)
        ax.set_title("Variables que más pesan en las decisiones del modelo")
        ax.set_xlabel("Aporte medio absoluto (puntos porcentuales de P(venenoso))")
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0, labelcolor=TIZA)
        ax.set_xlim(0, None)
        st.pyplot(_fig_ancho(fig))
        st.caption(META.get("global_importance_method", "Importancias nativas del modelo (feature_importances_)."))

# ================================================================== 5. EXPLORACIÓN DEL DATASET
elif NUMERO == "06":
    conteo = DF["class"].value_counts()
    especies_clase = pd.Series(CATALOGO.clases).value_counts()
    st.markdown(tema.cifras([
        ("Ejemplares", f"{len(DF):,}", f"{DF.shape[1]} columnas", ""),
        ("Especies", f"{len(CATALOGO.nombres)}", "353 ejemplares simulados por especie", ""),
        ("Venenosos", f"{conteo.get('p', 0) / len(DF):.1%}",
         f"{especies_clase.get('p', 0)} especies (incluye comestibilidad desconocida)", "mala"),
        ("Comestibles", f"{conteo.get('e', 0) / len(DF):.1%}", f"{especies_clase.get('e', 0)} especies", "buena"),
    ]), unsafe_allow_html=True)

    st.markdown(tema.seccion("Medidas por clase"), unsafe_allow_html=True)
    fig, ejes = plt.subplots(1, 3, figsize=(12, 3.4))
    for ax, col in zip(ejes, NUM_COLS):
        limite = DF[col].quantile(0.99)
        for clase, color, nombre in [("e", LIQUEN, "comestible"), ("p", AMANITA, "venenoso")]:
            datos = DF.loc[(DF["class"] == clase) & (DF[col] <= limite), col]
            ax.hist(datos, bins=45, color=color, alpha=0.10, histtype="stepfilled")
            ax.hist(datos, bins=45, color=color, lw=1.6, label=nombre, histtype="step")
        ax.set_title(ETIQUETAS_VARIABLES.get(col, col), fontsize=10)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
    ejes[0].legend(loc="upper right")
    st.pyplot(_fig_ancho(fig))
    st.caption("Histogramas superpuestos por clase (se omite el 1 % de valores más altos para que se vea la forma).")

    st.markdown(tema.seccion("Explorador de variables categóricas"), unsafe_allow_html=True)
    col_sel = st.selectbox(
        "Variable",
        CAT_COLS,
        format_func=lambda c: ETIQUETAS_VARIABLES.get(c, c),
        index=CAT_COLS.index("gill-color") if "gill-color" in CAT_COLS else 0,
    )
    tabla = pd.crosstab(DF[col_sel].fillna("sin dato"), DF["class"])
    tabla = tabla.loc[tabla.sum(axis=1).sort_values(ascending=False).index[:12]]
    mapa_sel = ETIQUETAS_CATEGORICAS.get(col_sel, {})
    etiquetas_cat = [mapa_sel.get(k, k) for k in tabla.index]
    fig, ax = plt.subplots(figsize=(9, 0.38 * len(tabla) + 1.3))
    comest = tabla.get("e", pd.Series(0, index=tabla.index)).to_numpy()
    venen = tabla.get("p", pd.Series(0, index=tabla.index)).to_numpy()
    ax.barh(etiquetas_cat, comest, color=LIQUEN, height=0.62, label="comestible")
    ax.barh(etiquetas_cat, venen, left=comest, color=AMANITA, height=0.62, label="venenoso")
    for i, (c, v) in enumerate(zip(comest, venen)):
        ax.text(c + v, i, f"  {v / (c + v):.0%} venenoso", va="center", fontsize=8, color=MUSGO)
    ax.invert_yaxis()
    ax.set_title(f"{ETIQUETAS_VARIABLES.get(col_sel, col_sel)} · ejemplares por categoría (top 12)")
    ax.set_xlabel("ejemplares")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, labelcolor=TIZA)
    ax.set_xlim(0, (comest + venen).max() * 1.22)
    ax.legend(loc="lower right")
    st.pyplot(_fig_ancho(fig))

# ================================================================== 6. FICHA TÉCNICA
elif NUMERO == "07":
    st.markdown(tema.seccion("Identidad visual"), unsafe_allow_html=True)
    c_logo1, c_logo2 = st.columns(2)
    c_logo1.markdown(
        f'<div style="background:{tema.SUPERFICIE};border:1px solid {LINEA};border-radius:14px;padding:28px">'
        f'{LOGOS["logotipo"].read_text(encoding="utf-8")}</div>', unsafe_allow_html=True)
    c_logo2.markdown(
        f'<div style="background:{TIZA};border-radius:14px;padding:28px">'
        f'{LOGOS["logotipo_claro"].read_text(encoding="utf-8")}</div>', unsafe_allow_html=True)
    st.caption("La esporada: la huella radial que dejan las láminas de un hongo. Archivos en `app/assets/`.")

    st.markdown(tema.seccion("Modelo"), unsafe_allow_html=True)
    st.markdown(tema.cifras([
        ("Modelo", META["model_name"], f"estimador {META.get('model_class', '—')}", ""),
        ("Umbral τ*", f"{TAU:.2f}", f"franja incierta {FRANJA[0]:.2f}–{FRANJA[1]:.2f}", ""),
        ("Variables", f"{len(TODAS_LAS_COLS)}", f"{META.get('num_features_encoded', '—')} tras One-Hot", ""),
    ], columnas=3), unsafe_allow_html=True)
    st.caption(f"Criterio de selección: {META['selection_criteria']}")

    if CALIBRACION:
        st.markdown(tema.seccion("Calibración del umbral τ*"), unsafe_allow_html=True)
        detalle_franja = ""
        if "margin_to_edible" in CALIBRACION:
            franja_cv = CALIBRACION["cv_out_of_fold_band"]
            detalle_franja = (
                f" Franja: {franja_cv['max_proba_edible']:.2f} – {franja_cv['min_proba_poisonous']:.2f}; margen de "
                f"{CALIBRACION['margin_to_edible']:.2f} hacia los comestibles y de "
                f"{CALIBRACION['margin_to_poisonous']:.2f} hacia los venenosos."
            )
        st.markdown(f"{CALIBRACION['method']}{detalle_franja}")
        st.markdown(
            f"Antes se usaba τ = {CALIBRACION['single_split_threshold']:.2f}: el umbral más alto con FN = 0 en un "
            "único split de validación. Dependía del split (falla en un fold) y quedaba en el borde de la franja "
            "segura, sin margen."
        )
        prueba = CALIBRACION["test_at_threshold"]
        columnas_fold = {
            "Fold": range(1, len(CALIBRACION["fold_thresholds"]) + 1),
            "Umbral más alto con FN = 0": CALIBRACION["fold_thresholds"],
        }
        if "fold_fn_at_threshold" in CALIBRACION:
            columnas_fold[f"FN con τ* = {TAU:.2f}"] = CALIBRACION["fold_fn_at_threshold"]
            columnas_fold[f"FP con τ* = {TAU:.2f}"] = CALIBRACION["fold_fp_at_threshold"]
        columnas_fold[f"FN con τ = {CALIBRACION['single_split_threshold']:.2f}"] = (
            CALIBRACION["fold_fn_at_single_split_threshold"]
        )
        st.dataframe(pd.DataFrame(columnas_fold), width="stretch", hide_index=True)
        st.caption(
            f"Verificación en test con τ* = {TAU:.2f}: FN = {prueba['fn']}, FP = {prueba['fp']}, "
            f"precisión = {prueba['precision']:.2%}, recall = {prueba['recall']:.2%}. En test, el venenoso "
            f"con menor probabilidad tuvo P = {CALIBRACION['test_min_proba_poisonous']:.2f} y el comestible "
            f"con mayor probabilidad, P = {CALIBRACION['test_max_proba_edible']:.2f}."
        )

    with st.expander("Metadatos completos (model_metadata.json)"):
        st.json(META)

    st.markdown(tema.seccion("Funcionalidades de la aplicación"), unsafe_allow_html=True)
    st.markdown(
        "\n".join(
            f"{i}. {f}"
            for i, f in enumerate(
                [
                    "Predicción individual interactiva con las 20 características morfológicas.",
                    "Predicción por lotes desde archivo CSV con validación de columnas.",
                    "Probabilidad de toxicidad en tiempo real (predict_proba del pipeline).",
                    "Umbral de riesgo cero τ* aplicado al dictamen y explorable con un deslizador.",
                    "Desglose local de la probabilidad por variable (contribuciones nativas del Random Forest).",
                    "Simulador «¿qué pasaría si…?» para cualquier variable del espécimen.",
                    "Importancia global precalculada (sin cálculo al arrancar la app).",
                    "Dashboard de métricas del modelo desplegado en el conjunto de test.",
                    "Comparativa de los 4 modelos entrenados (LR, Random Forest, XGBoost, MLP).",
                    "Matriz de confusión del modelo desplegado.",
                    "Exploración del dataset con visualizaciones (INFOVIZ).",
                    "Descarga de resultados por lotes y plantilla CSV de ejemplo.",
                    "Ficha técnica y documentación matemática embebida.",
                    "Validación cruzada estratificada k-fold sin fuga de datos (pipeline por fold, test excluido).",
                    "Detector de catálogo: dictamen solo si el espécimen encaja 20/20 en una especie conocida.",
                    "Dictamen en tres estados (comestible, venenoso, incierto) con franja de incertidumbre.",
                    "Validación agrupada por especie para medir el rendimiento con especies nuevas.",
                ],
                start=1,
            )
        )
    )

    st.divider()
    st.markdown("## Documentación matemática de los modelos")
    if FORMULAS_PATH.exists():
        texto = FORMULAS_PATH.read_text(encoding="utf-8")
        import re

        partes = re.split(r"!\[([^\]]*)\]\(([^)]+)\)", texto)
        for i, parte in enumerate(partes):
            if i % 3 == 0:
                st.markdown(parte)
            elif i % 3 == 1:
                alt = parte
            else:
                ruta = (FORMULAS_PATH.parent / parte).resolve()
                if ruta.exists():
                    st.image(str(ruta), caption=alt, width="stretch")
    else:
        st.warning("No se encontró docs/formulas_matematicas_modelos.md.")

    st.divider()
    st.caption("Demo del Hito 2 — Electiva en Ingeniería Aplicada II (Data Science Fundamentals).")
