from typing import Dict, Any, Tuple
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import pandas as pd


def build_preprocessing_pipeline(
    numeric_features: list, 
    categorical_features: list
) -> ColumnTransformer:
    """
    Construye un transformador de columnas con imputación y escalado/one-hot encoding.
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )
    return preprocessor


def get_baseline_models() -> Dict[str, Any]:
    """
    Retorna un diccionario con modelos base para experimentación.
    """
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=10),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    }


def create_model_pipeline(
    model, 
    numeric_features: list, 
    categorical_features: list
) -> Pipeline:
    """
    Combina el preprocesamiento con el estimador final en un único pipeline de scikit-learn.
    """
    preprocessor = build_preprocessing_pipeline(numeric_features, categorical_features)
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])
