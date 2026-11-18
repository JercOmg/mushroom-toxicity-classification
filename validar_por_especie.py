# -*- coding: utf-8 -*-
"""Validación cruzada agrupada por especie y validación del detector de catálogo.

La validación cruzada aleatoria (notebook 03 y recalibrar_umbral_cv.py) reparte ejemplares de
la misma especie entre entrenamiento y validación, así que mide si el modelo REconoce especies
que ya vio. Este script mide lo que pasa con especies NUEVAS:

1. StratifiedGroupKFold (k=5) con la especie como grupo: ninguna especie aparece a la vez en
   entrenamiento y validación. Se guarda la probabilidad fuera de fold de cada ejemplar.
2. Detector de catálogo (src/especies.py): para cada ejemplar se excluye su propia especie y se
   comprueba si aun así encaja 20/20 en alguna otra. Así se estima cuántos especímenes de
   especies desconocidas recibirían dictamen y con qué tasa de error.
3. Escribe los resultados en app/models/model_metadata.json (clave «species_validation»).

Ejecutar después de notebooks/03_model_training.ipynb y recalibrar_umbral_cv.py:
    python validar_por_especie.py
"""
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from src.data_loader import NUMERIC_COLUMNS
from src.especies import CatalogoEspecies, asignar_especie
from src.models import create_model_pipeline, get_candidate_models

# Sin .resolve(): en Windows, resolver la ruta real puede superar los 260 caracteres.
ROOT = Path(__file__).absolute().parent
METADATA_PATH = ROOT / "app" / "models" / "model_metadata.json"
K = 5
RANDOM_STATE = 42


def errores(y: np.ndarray, proba: np.ndarray, tau: float, mascara=None) -> dict:
    m = np.ones(len(y), bool) if mascara is None else mascara
    pred = proba >= tau
    return {
        "n": int(m.sum()),
        "accuracy": float((pred == y)[m].mean()) if m.any() else None,
        "fn": int((~pred & (y == 1) & m).sum()),
        "fp": int((pred & (y == 0) & m).sum()),
        "fn_rate": float((~pred & (y == 1) & m).sum() / max(((y == 1) & m).sum(), 1)),
    }


def main() -> None:
    meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    tau = float(meta["optimal_threshold"])
    modelo = meta["model_name"]

    secundario = pd.read_csv(ROOT / "data" / "raw" / "secondary_data.csv", sep=";")
    primario = pd.read_csv(ROOT / "data" / "raw" / "primary_data.csv", sep=";")
    X = secundario.drop(columns="class")
    y = secundario["class"].map({"e": 0, "p": 1}).to_numpy()
    especie = asignar_especie(len(secundario))
    categoricas = [c for c in X.columns if c not in NUMERIC_COLUMNS]

    print(f"Validación cruzada {K}-fold agrupada por especie ({especie.max() + 1} especies) — {modelo}...")
    oof = np.zeros(len(X))
    for tr, va in StratifiedGroupKFold(K, shuffle=True, random_state=RANDOM_STATE).split(X, y, especie):
        pipe = create_model_pipeline(clone(get_candidate_models()[modelo]), NUMERIC_COLUMNS, categoricas)
        pipe.fit(X.iloc[tr], y[tr])
        oof[va] = pipe.predict_proba(X.iloc[va])[:, 1]

    por_especie = pd.DataFrame({"g": especie, "y": y, "p": oof}).groupby("g").agg(y=("y", "first"), p=("p", "mean"))
    especies_fn = int(((por_especie.p < tau) & (por_especie.y == 1)).sum())
    especies_fp = int(((por_especie.p >= tau) & (por_especie.y == 0)).sum())
    global_ = errores(y, oof, tau)
    print(f"  Especies nuevas: accuracy = {global_['accuracy']:.3f}, ROC-AUC = {roc_auc_score(y, oof):.3f}, "
          f"venenosos tomados por comestibles = {global_['fn_rate']:.1%}")

    catalogo = CatalogoEspecies(secundario, primario, NUMERIC_COLUMNS, categoricas)
    total = len(NUMERIC_COLUMNS) + len(categoricas)
    comp = catalogo.compatibilidad(X)
    propia = comp[np.arange(len(X)), especie]
    comp[np.arange(len(X)), especie] = -1
    mejor_ajena = comp.max(axis=1)
    pasan = mejor_ajena >= total
    filtrado = errores(y, oof, tau, pasan)
    print(f"  Detector: {pasan.mean():.2%} de los ejemplares de especies desconocidas pasarían el filtro; "
          f"entre ellos FN = {filtrado['fn']}, accuracy = {filtrado['accuracy']}")

    meta["species_validation"] = {
        "method": (
            f"StratifiedGroupKFold k={K} con la especie como grupo (173 especies × 353 ejemplares; "
            "especie = fila // 353, verificado contra primary_data.csv). Ninguna especie se usa a la vez "
            "para entrenar y validar."
        ),
        "n_species": int(especie.max() + 1),
        "threshold": tau,
        "unseen_species": {
            **global_,
            "roc_auc": float(roc_auc_score(y, oof)),
            "species_poisonous_missed": especies_fn,
            "species_poisonous_total": int((por_especie.y == 1).sum()),
            "species_edible_discarded": especies_fp,
            "species_edible_total": int((por_especie.y == 0).sum()),
        },
        "catalog_detector": {
            "rule": f"Dictamen solo si el espécimen encaja {total}/{total} en el perfil de alguna especie conocida.",
            "numeric_margin": 0.10,
            "known_rows_in_catalog": float((propia == total).mean()),
            "unseen_rows_passing": float(pasan.mean()),
            "unseen_best_match_distribution": {int(k): int(v) for k, v in
                                               pd.Series(mejor_ajena).value_counts().sort_index().items()},
            "errors_on_passing_rows": filtrado,
        },
        "computed_at": datetime.now().isoformat(timespec="seconds"),
    }
    METADATA_PATH.write_text(json.dumps(meta, indent=4, ensure_ascii=False), encoding="utf-8")
    print(f"✓ {METADATA_PATH.relative_to(ROOT)} actualizado (species_validation)")


if __name__ == "__main__":
    main()
