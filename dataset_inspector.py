from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURACIÓN
# ============================================================

DATASET_PATH = Path(
    "MushroomDataset/MushroomDataset/secondary_data.csv"
)

RESULTADOS_PATH = Path("resultados")

TARGET = "class"

COLUMNAS_NUMERICAS = [
    "cap-diameter",
    "stem-height",
    "stem-width",
]


# Crear carpeta de resultados
RESULTADOS_PATH.mkdir(
    exist_ok=True
)


# ============================================================
# ENCABEZADO
# ============================================================

print("=" * 70)
print("        INSPECCIÓN DEL SECONDARY MUSHROOM DATASET")
print("=" * 70)


# ============================================================
# CONTROL 1: EXISTENCIA DEL ARCHIVO
# ============================================================

print("\n" + "=" * 70)
print("CONTROL 1 - EXISTENCIA DEL ARCHIVO")
print("=" * 70)

print("\nRuta:")
print(DATASET_PATH.resolve())

if not DATASET_PATH.exists():

    print("\nERROR: El archivo no existe.")
    exit()

print("\nOK - Archivo encontrado correctamente.")

tamano_mb = (
    DATASET_PATH.stat().st_size
    / (1024 * 1024)
)

print(
    f"Tamaño del archivo: "
    f"{tamano_mb:.2f} MB"
)


# ============================================================
# CONTROL 2: CARGA E INTEGRIDAD DEL CSV
# ============================================================

print("\n" + "=" * 70)
print("CONTROL 2 - CARGA E INTEGRIDAD")
print("=" * 70)

try:

    df = pd.read_csv(
        DATASET_PATH,
        sep=";"
    )

except Exception as error:

    print(
        "\nERROR al leer el dataset:"
    )

    print(error)

    exit()


print("\nOK - Dataset cargado correctamente.")

print(
    f"Registros: {df.shape[0]}"
)

print(
    f"Columnas: {df.shape[1]}"
)


# ============================================================
# CONTROL 3: ESTRUCTURA DEL DATASET
# ============================================================

print("\n" + "=" * 70)
print("CONTROL 3 - ESTRUCTURA")
print("=" * 70)

print("\nColumnas encontradas:\n")

for numero, columna in enumerate(
    df.columns,
    start=1
):

    print(
        f"{numero:02d}. {columna}"
    )


# ============================================================
# CONTROL 4: TIPOS DE DATOS
# ============================================================

print("\n" + "=" * 70)
print("CONTROL 4 - TIPOS DE DATOS")
print("=" * 70)

tipos = pd.DataFrame({
    "columna": df.columns,
    "tipo": [
        str(tipo)
        for tipo in df.dtypes
    ]
})

print(
    tipos.to_string(
        index=False
    )
)

tipos.to_csv(
    RESULTADOS_PATH
    / "01_tipos_datos.csv",
    index=False
)


# ============================================================
# CONTROL 5: VALORES NULOS
# ============================================================

print("\n" + "=" * 70)
print("CONTROL 5 - VALORES NULOS")
print("=" * 70)

nulos = df.isnull().sum()

porcentaje_nulos = (
    df.isnull().mean()
    * 100
)

reporte_nulos = pd.DataFrame({
    "columna": df.columns,
    "cantidad_nulos": nulos.values,
    "porcentaje_nulos":
        porcentaje_nulos.values
})

reporte_nulos = (
    reporte_nulos
    .sort_values(
        "cantidad_nulos",
        ascending=False
    )
)

print(
    reporte_nulos.to_string(
        index=False
    )
)

reporte_nulos.to_csv(
    RESULTADOS_PATH
    / "02_valores_nulos.csv",
    index=False
)


# ============================================================
# CONTROL 6: CELDAS VACÍAS
# ============================================================

print("\n" + "=" * 70)
print("CONTROL 6 - CELDAS VACÍAS")
print("=" * 70)

reporte_vacios = []

for columna in df.columns:

    if df[columna].dtype == "object":

        cantidad = (
            df[columna]
            .fillna("")
            .astype(str)
            .str.strip()
            .eq("")
            .sum()
        )

    else:

        cantidad = 0

    reporte_vacios.append({
        "columna": columna,
        "celdas_vacias": cantidad
    })


df_vacios = pd.DataFrame(
    reporte_vacios
)

print(
    df_vacios.to_string(
        index=False
    )
)

df_vacios.to_csv(
    RESULTADOS_PATH
    / "03_celdas_vacias.csv",
    index=False
)


# ============================================================
# CONTROL 7: REGISTROS DUPLICADOS
# ============================================================

print("\n" + "=" * 70)
print("CONTROL 7 - REGISTROS DUPLICADOS")
print("=" * 70)

cantidad_duplicados = (
    df.duplicated().sum()
)

print(
    "\nCantidad de registros "
    f"duplicados: {cantidad_duplicados}"
)

if cantidad_duplicados > 0:

    duplicados = df[
        df.duplicated(
            keep=False
        )
    ]

    duplicados.to_csv(
        RESULTADOS_PATH
        / "04_registros_duplicados.csv",
        index=False
    )

    print(
        "\nSe guardaron en:"
    )

    print(
        "resultados/"
        "04_registros_duplicados.csv"
    )


# ============================================================
# CONTROL 8: VARIABLE OBJETIVO
# ============================================================

print("\n" + "=" * 70)
print("CONTROL 8 - BALANCE DE CLASES")
print("=" * 70)

if TARGET not in df.columns:

    print(
        "\nERROR: No existe "
        "la variable objetivo."
    )

else:

    conteo = (
        df[TARGET]
        .value_counts(
            dropna=False
        )
    )

    porcentaje = (
        df[TARGET]
        .value_counts(
            normalize=True,
            dropna=False
        )
        * 100
    )

    balance = pd.DataFrame({
        "cantidad": conteo,
        "porcentaje": porcentaje
    })

    print("\n")
    print(balance)

    balance.to_csv(
        RESULTADOS_PATH
        / "05_balance_clases.csv"
    )


# ============================================================
# CONTROL 9: VARIABLES NUMÉRICAS
# ============================================================

print("\n" + "=" * 70)
print("CONTROL 9 - VARIABLES NUMÉRICAS")
print("=" * 70)

for columna in COLUMNAS_NUMERICAS:

    if columna not in df.columns:
        continue

    df[columna] = pd.to_numeric(
        df[columna],
        errors="coerce"
    )


estadisticas = (
    df[COLUMNAS_NUMERICAS]
    .describe()
    .T
)

print("\n")
print(estadisticas)

estadisticas.to_csv(
    RESULTADOS_PATH
    / "06_estadisticas_numericas.csv"
)


# ============================================================
# CONTROL 10: VALORES ATÍPICOS
# MÉTODO IQR
# ============================================================

print("\n" + "=" * 70)
print("CONTROL 10 - POSIBLES VALORES ATÍPICOS")
print("=" * 70)

reporte_outliers = []

for columna in COLUMNAS_NUMERICAS:

    serie = (
        df[columna]
        .dropna()
    )

    q1 = serie.quantile(
        0.25
    )

    q3 = serie.quantile(
        0.75
    )

    iqr = q3 - q1

    limite_inferior = (
        q1
        - 1.5 * iqr
    )

    limite_superior = (
        q3
        + 1.5 * iqr
    )

    outliers = serie[
        (
            serie
            < limite_inferior
        )
        |
        (
            serie
            > limite_superior
        )
    ]

    reporte_outliers.append({
        "variable":
            columna,

        "q1":
            q1,

        "q3":
            q3,

        "iqr":
            iqr,

        "limite_inferior":
            limite_inferior,

        "limite_superior":
            limite_superior,

        "cantidad_outliers":
            len(outliers),

        "porcentaje":
            (
                len(outliers)
                / len(serie)
            )
            * 100
    })


df_outliers = pd.DataFrame(
    reporte_outliers
)

print("\n")

print(
    df_outliers.to_string(
        index=False
    )
)

df_outliers.to_csv(
    RESULTADOS_PATH
    / "07_outliers.csv",
    index=False
)


# ============================================================
# CONTROL 11: VARIABLES CATEGÓRICAS
# ============================================================

print("\n" + "=" * 70)
print("CONTROL 11 - VARIABLES CATEGÓRICAS")
print("=" * 70)

columnas_categoricas = [
    columna
    for columna in df.columns
    if columna
    not in COLUMNAS_NUMERICAS
]


reporte_categorias = []

for columna in columnas_categoricas:

    valores = (
        df[columna]
        .value_counts(
            dropna=False
        )
    )

    print(
        f"\n--- {columna} ---"
    )

    print(valores)

    for valor, cantidad in valores.items():

        reporte_categorias.append({
            "columna":
                columna,

            "valor":
                valor,

            "cantidad":
                cantidad
        })


pd.DataFrame(
    reporte_categorias
).to_csv(
    RESULTADOS_PATH
    / "08_categorias.csv",
    index=False
)


# ============================================================
# CONTROL 12: VALORES ÚNICOS
# ============================================================

print("\n" + "=" * 70)
print("CONTROL 12 - CANTIDAD DE VALORES ÚNICOS")
print("=" * 70)

valores_unicos = []

for columna in df.columns:

    cantidad = (
        df[columna]
        .nunique(
            dropna=True
        )
    )

    valores_unicos.append({
        "columna":
            columna,

        "valores_unicos":
            cantidad
    })


df_unicos = pd.DataFrame(
    valores_unicos
)

print(
    df_unicos.to_string(
        index=False
    )
)

df_unicos.to_csv(
    RESULTADOS_PATH
    / "09_valores_unicos.csv",
    index=False
)


# ============================================================
# CONTROL 13: MUESTRA DE DATOS
# ============================================================

print("\n" + "=" * 70)
print("CONTROL 13 - MUESTRA DE REGISTROS")
print("=" * 70)

print(
    df.head(10)
    .to_string(index=False)
)


# ============================================================
# ESTANDARIZACIÓN BÁSICA DEL DATASET
# ============================================================

print("\n" + "=" * 70)
print("ESTANDARIZACIÓN DEL DATASET")
print("=" * 70)

df_estandarizado = (
    df.copy()
)


# ------------------------------------------------------------
# Eliminar espacios al principio y al final
# ------------------------------------------------------------

for columna in columnas_categoricas:

    df_estandarizado[columna] = (
        df_estandarizado[columna]
        .apply(
            lambda valor:
            valor.strip().lower()
            if isinstance(
                valor,
                str
            )
            else valor
        )
    )


# ------------------------------------------------------------
# Asegurar tipos numéricos
# ------------------------------------------------------------

for columna in COLUMNAS_NUMERICAS:

    df_estandarizado[columna] = (
        pd.to_numeric(
            df_estandarizado[
                columna
            ],
            errors="coerce"
        )
    )


# ------------------------------------------------------------
# Exportar dataset estandarizado
# ------------------------------------------------------------

ARCHIVO_ESTANDARIZADO = (
    RESULTADOS_PATH
    / "secondary_mushroom_estandarizado.csv"
)

df_estandarizado.to_csv(
    ARCHIVO_ESTANDARIZADO,
    index=False
)

print(
    "\nDataset estandarizado guardado:"
)

print(
    ARCHIVO_ESTANDARIZADO.resolve()
)


# ============================================================
# GRÁFICA 1: BALANCE DE CLASES
# ============================================================

if TARGET in df.columns:

    conteo_clases = (
        df[TARGET]
        .value_counts()
    )

    plt.figure(
        figsize=(7, 5)
    )

    plt.bar(
        conteo_clases.index
        .astype(str),

        conteo_clases.values
    )

    plt.title(
        "Distribución de clases"
    )

    plt.xlabel(
        "Clase"
    )

    plt.ylabel(
        "Cantidad de registros"
    )

    plt.tight_layout()

    plt.savefig(
        RESULTADOS_PATH
        / "10_balance_clases.png",
        dpi=150
    )

    plt.close()


# ============================================================
# GRÁFICA 2: DISTRIBUCIONES NUMÉRICAS
# ============================================================

for columna in COLUMNAS_NUMERICAS:

    plt.figure(
        figsize=(7, 5)
    )

    plt.hist(
        df[columna]
        .dropna(),
        bins=30
    )

    plt.title(
        f"Distribución de {columna}"
    )

    plt.xlabel(
        columna
    )

    plt.ylabel(
        "Frecuencia"
    )

    plt.tight_layout()

    plt.savefig(
        RESULTADOS_PATH
        / f"11_distribucion_{columna}.png",
        dpi=150
    )

    plt.close()


# ============================================================
# GRÁFICA 3: BOX PLOTS
# ============================================================

for columna in COLUMNAS_NUMERICAS:

    plt.figure(
        figsize=(7, 5)
    )

    plt.boxplot(
        df[columna]
        .dropna(),
        vert=True
    )

    plt.title(
        f"Boxplot de {columna}"
    )

    plt.ylabel(
        columna
    )

    plt.tight_layout()

    plt.savefig(
        RESULTADOS_PATH
        / f"12_boxplot_{columna}.png",
        dpi=150
    )

    plt.close()


# ============================================================
# RESUMEN FINAL
# ============================================================

print("\n" + "=" * 70)
print("RESUMEN FINAL")
print("=" * 70)

print(
    f"\nRegistros totales: "
    f"{len(df)}"
)

print(
    f"Variables totales: "
    f"{len(df.columns)}"
)

print(
    f"Variables numéricas: "
    f"{len(COLUMNAS_NUMERICAS)}"
)

print(
    "Variables categóricas: "
    f"{len(columnas_categoricas)}"
)

print(
    f"Registros duplicados: "
    f"{cantidad_duplicados}"
)

print(
    "Total de valores nulos: "
    f"{df.isnull().sum().sum()}"
)

print("\nArchivos generados en:")

print(
    RESULTADOS_PATH.resolve()
)

print(
    "\nINSPECCIÓN FINALIZADA CORRECTAMENTE"
)