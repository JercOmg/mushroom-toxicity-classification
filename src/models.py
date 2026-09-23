from typing import Dict, Any, Tuple
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
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


def get_candidate_models() -> Dict[str, Any]:
    """
    Retorna los 4 modelos de Machine Learning requeridos para la propuesta técnica:
    1. Regresión Logística (LogisticRegression)
    2. Random Forest (RandomForestClassifier)
    3. XGBoost (XGBClassifier)
    4. Red Neuronal Perceptrón Multicapa (MLPClassifier)
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, 
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, 
            random_state=42, 
            n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100, 
            max_depth=6, 
            learning_rate=0.1, 
            random_state=42, 
            eval_metric="logloss", 
            n_jobs=-1
        ),
        "MLP Neural Network": MLPClassifier(
            hidden_layer_sizes=(64, 32), 
            max_iter=200, 
            random_state=42, 
            early_stopping=True
        )
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
