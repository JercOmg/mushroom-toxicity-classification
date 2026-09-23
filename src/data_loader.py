from pathlib import Path
from typing import Tuple, Optional
import pandas as pd
from sklearn.model_selection import train_test_split

DEFAULT_RAW_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "secondary_data.csv"
DEFAULT_PROCESSED_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "secondary_mushroom_estandarizado.csv"

NUMERIC_COLUMNS = ["cap-diameter", "stem-height", "stem-width"]
TARGET_COLUMN = "class"


def load_raw_data(filepath: Optional[Path] = None, sep: str = ";") -> pd.DataFrame:
    """
    Carga el dataset secundario en su formato crudo.
    """
    path = Path(filepath) if filepath else DEFAULT_RAW_PATH
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de datos en: {path}")
    return pd.read_csv(path, sep=sep)


def load_processed_data(filepath: Optional[Path] = None) -> pd.DataFrame:
    """
    Carga el dataset ya estandarizado y procesado.
    """
    path = Path(filepath) if filepath else DEFAULT_PROCESSED_PATH
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo procesado en: {path}")
    return pd.read_csv(path)


def split_features_target(
    df: pd.DataFrame, 
    target_col: str = TARGET_COLUMN
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separa las características (X) de la variable objetivo (y).
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def get_train_test_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    target_col: str = TARGET_COLUMN
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Divide el dataset en conjuntos de entrenamiento y prueba estratificados.
    """
    X, y = split_features_target(df, target_col=target_col)
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
