from pathlib import Path
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


def plot_confusion_matrix(
    y_true, 
    y_pred, 
    labels=None, 
    title: str = "Matriz de Confusión", 
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
        cmap="Blues",
        xticklabels=labels if labels is not None else ["e", "p"],
        yticklabels=labels if labels is not None else ["e", "p"]
    )
    plt.title(title)
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.show()


def evaluate_predictions(y_true, y_pred) -> Dict[str, Any]:
    """
    Genera un diccionario con el reporte de métricas de clasificación.
    """
    return classification_report(y_true, y_pred, output_dict=True)
