from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score
)


def plot_confusion_matrix(
    y_true, 
    y_pred, 
    labels=None, 
    title: str = "Matriz de Confusión", 
    cmap: str = "Blues",
    save_path: Optional[Path] = None
):
    """
    Grafica la matriz de confusión con mapa de calor (heatmap).
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap=cmap,
        xticklabels=labels if labels is not None else ["e", "p"],
        yticklabels=labels if labels is not None else ["e", "p"]
    )
    plt.title(title, fontsize=12, fontweight="bold")
    plt.xlabel("Predicción", fontsize=11)
    plt.ylabel("Real", fontsize=11)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def evaluate_predictions(y_true, y_pred) -> Dict[str, Any]:
    """
    Genera un diccionario con el reporte de métricas de clasificación.
    """
    return classification_report(y_true, y_pred, output_dict=True)


def evaluate_model_performance(
    y_true, 
    y_pred, 
    model_name: str = "Modelo"
) -> Dict[str, Any]:
    """
    Calcula las métricas clave orientadas a la evaluación binaria (clase positiva = 1 / venenoso).
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall (Poisonous)": recall_score(y_true, y_pred, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0),
        "FN (Falsos Negativos)": int(fn),
        "FP (Falsos Positivos)": int(fp),
        "TP (Verdaderos Positivos)": int(tp),
        "TN (Verdaderos Negativos)": int(tn)
    }


def evaluate_thresholds(
    y_true: np.ndarray, 
    y_probs_poisonous: np.ndarray, 
    thresholds: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """
    Evalúa una grilla de umbrales para clasificar la probabilidad de la clase venenosa.
    Retorna un DataFrame con métricas (Recall, Precision, F1, FN, FP, TP, TN) para cada umbral.
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.50, 50)

    records = []
    for th in thresholds:
        y_pred = (y_probs_poisonous >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        records.append({
            "threshold": round(float(th), 4),
            "FN": int(fn),
            "FP": int(fp),
            "TP": int(tp),
            "TN": int(tn),
            "Recall (Poisonous)": recall,
            "Precision": precision,
            "F1-Score": f1
        })

    return pd.DataFrame(records)


def find_zero_fn_threshold(
    threshold_df: pd.DataFrame
) -> Tuple[float, pd.Series]:
    """
    Encuentra el umbral óptimo más alto que garantiza 0 Falsos Negativos (FN = 0),
    maximizando la precisión dentro de la zona de riesgo cero.
    """
    zero_fn = threshold_df[threshold_df["FN"] == 0]
    if not zero_fn.empty:
        # Priorizar el mayor umbral (menos FP) y mayor precisión
        best_row = zero_fn.sort_values(
            by=["threshold", "Precision"], 
            ascending=[False, False]
        ).iloc[0]
        return float(best_row["threshold"]), best_row
    else:
        # Si ningún umbral logra FN=0, tomar el que minimiza FN
        best_row = threshold_df.sort_values(
            by=["FN", "threshold"], 
            ascending=[True, False]
        ).iloc[0]
        return float(best_row["threshold"]), best_row


def plot_threshold_tuning_curve(
    threshold_df: pd.DataFrame, 
    optimal_threshold: Optional[float] = None,
    title: str = "Ajuste de Umbral: Metodología de Riesgo Cero",
    save_path: Optional[Path] = None
):
    """
    Grafica la evolución de Falsos Negativos (FN), Recall y Precisión en función del umbral.
    """
    fig, ax1 = plt.subplots(figsize=(9, 5))

    ax1.set_xlabel("Umbral de Decisión (Threshold)", fontsize=11)
    ax1.set_ylabel("Métricas (Recall / Precision)", color="tab:blue", fontsize=11)
    line1 = ax1.plot(threshold_df["threshold"], threshold_df["Recall (Poisonous)"], label="Recall (Venenoso)", color="tab:green", lw=2.5)
    line2 = ax1.plot(threshold_df["threshold"], threshold_df["Precision"], label="Precision", color="tab:blue", lw=2, linestyle="--")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_ylim(-0.05, 1.05)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Falsos Negativos (Hongos venenosos ingeridos)", color="tab:red", fontsize=11)
    line3 = ax2.plot(threshold_df["threshold"], threshold_df["FN"], label="Falsos Negativos (FN)", color="tab:red", lw=2, linestyle="-.")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    if optimal_threshold is not None:
        ax1.axvline(optimal_threshold, color="black", linestyle=":", lw=2, label=f"Umbral Óptimo ({optimal_threshold:.3f})")

    # Unir leyendas
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    if optimal_threshold is not None:
        lines.append(plt.Line2D([0], [0], color="black", linestyle=":", lw=2))
        labels.append(f"Umbral Óptimo ({optimal_threshold:.3f})")

    ax1.legend(lines, labels, loc="center left", framealpha=0.9)
    plt.title(title, fontsize=12, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
