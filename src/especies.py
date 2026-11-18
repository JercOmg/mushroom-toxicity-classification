# -*- coding: utf-8 -*-
"""Catálogo de especies y detector de especímenes fuera del catálogo.

El *Secondary Mushroom Dataset* se simuló con 353 ejemplares por especie, en el mismo
orden que las 173 especies de `primary_data.csv` (fila i -> especie i // 353). La
validación cruzada agrupada por especie muestra que el clasificador acierta casi siempre
con especies que ya vio, pero se acerca al azar con especies nuevas. Por eso la app solo
emite un dictamen cuando el espécimen encaja en el perfil de una especie conocida.

Perfil de una especie:
* variables nominales: conjunto de valores observados en sus 353 ejemplares
  (incluido «sin dato» cuando la variable no aplica a esa especie);
* variables métricas: rango observado [mín, máx] ampliado con un margen relativo.
"""
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

FILAS_POR_ESPECIE = 353
MARGEN_NUMERICO = 0.10  # ±10 % del ancho del rango observado
SIN_DATO = "__sin_dato__"


def asignar_especie(n_filas: int) -> np.ndarray:
    return np.arange(n_filas) // FILAS_POR_ESPECIE


class CatalogoEspecies:
    def __init__(self, secundario: pd.DataFrame, primario: pd.DataFrame,
                 columnas_numericas: list, columnas_categoricas: list):
        self.num = list(columnas_numericas)
        self.cat = list(columnas_categoricas)
        especie = asignar_especie(len(secundario))
        n_especies = especie.max() + 1
        if len(primario) != n_especies:
            raise ValueError("primary_data.csv no coincide con el número de bloques de especie")

        self.nombres = primario["name"].tolist()
        self.familias = primario["family"].tolist()
        self.clases = primario["class"].tolist()

        grupos = secundario.groupby(especie)
        minimo = grupos[self.num].min()
        maximo = grupos[self.num].max()
        ancho = (maximo - minimo).clip(lower=0.5)
        self.num_min = (minimo - MARGEN_NUMERICO * ancho).to_numpy()
        self.num_max = (maximo + MARGEN_NUMERICO * ancho).to_numpy()

        # Para cada variable nominal: matriz booleana especies × valores posibles.
        self.vocab, self.permitidos = {}, {}
        for c in self.cat:
            valores = secundario[c].fillna(SIN_DATO)
            vocab = sorted(valores.unique())
            indice = {v: i for i, v in enumerate(vocab)}
            matriz = np.zeros((n_especies, len(vocab)), dtype=bool)
            matriz[especie, valores.map(indice).to_numpy()] = True
            self.vocab[c], self.permitidos[c] = indice, matriz

    # ------------------------------------------------------------------ coincidencias
    def compatibilidad(self, X: pd.DataFrame) -> np.ndarray:
        """Matriz (n_filas × n_especies) con cuántas de las 20 variables encajan en cada especie."""
        total = np.zeros((len(X), len(self.nombres)), dtype=np.int16)
        for j, c in enumerate(self.num):
            v = X[c].to_numpy(dtype=float)[:, None]
            total += ((v >= self.num_min[None, :, j]) & (v <= self.num_max[None, :, j])).astype(np.int16)
        for c in self.cat:
            idx = X[c].fillna(SIN_DATO).map(self.vocab[c]).to_numpy()
            conocido = ~pd.isna(idx)
            fila = np.zeros((len(X), len(self.nombres)), dtype=bool)
            fila[conocido] = self.permitidos[c][:, idx[conocido].astype(int)].T
            total += fila.astype(np.int16)
        return total

    def diagnostico(self, fila: dict, excluir: Optional[int] = None) -> dict:
        """Especie más parecida a un espécimen y qué variables no encajan en ella."""
        X = pd.DataFrame([fila])[self.num + self.cat]
        comp = self.compatibilidad(X)[0].astype(int)
        if excluir is not None:
            comp[excluir] = -1
        mejor = int(comp.argmax())
        fuera = []
        for j, c in enumerate(self.num):
            if not (self.num_min[mejor, j] <= float(fila[c]) <= self.num_max[mejor, j]):
                fuera.append(c)
        for c in self.cat:
            v = SIN_DATO if pd.isna(fila.get(c)) else fila.get(c)
            i = self.vocab[c].get(v)
            if i is None or not self.permitidos[c][mejor, i]:
                fuera.append(c)
        empatadas = int((comp == comp[mejor]).sum())
        return {
            "indice": mejor,
            "nombre": self.nombres[mejor],
            "familia": self.familias[mejor],
            "clase": self.clases[mejor],
            "coinciden": int(comp[mejor]),
            "total": len(self.num) + len(self.cat),
            "variables_fuera": fuera,
            "especies_empatadas": empatadas,
            "en_catalogo": int(comp[mejor]) == len(self.num) + len(self.cat),
        }

    def rango_numerico(self, especie: int, columna: str) -> tuple:
        j = self.num.index(columna)
        return float(self.num_min[especie, j]), float(self.num_max[especie, j])

    def valores_permitidos(self, especie: int, columna: str) -> list:
        inverso = {i: v for v, i in self.vocab[columna].items()}
        return [inverso[i] for i in np.flatnonzero(self.permitidos[columna][especie])]


def cargar_catalogo(raiz: Path, columnas_numericas: list, columnas_categoricas: list) -> CatalogoEspecies:
    secundario = pd.read_csv(raiz / "data" / "raw" / "secondary_data.csv", sep=";")
    primario = pd.read_csv(raiz / "data" / "raw" / "primary_data.csv", sep=";")
    return CatalogoEspecies(secundario, primario, columnas_numericas, columnas_categoricas)
