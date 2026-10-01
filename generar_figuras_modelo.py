# -*- coding: utf-8 -*-
"""Generador de esquemas matemáticos del documento docs/formulas_matematicas_modelos.md
(criterio de rúbrica: Modelo Matemático — ecuaciones explicadas y gráficos descriptivos).

Produce en docs/figuras_modelo_matematico/:
  - esquema_sigmoide_bce.png   (Regresión Logística: sigmoide + Binary Cross-Entropy)
  - esquema_arbol_bosque.png   (Random Forest: árbol CART + agregación bagging)
  - esquema_boosting.png       (XGBoost: modelo aditivo secuencial)
  - esquema_mlp.png            (MLP: arquitectura 128 -> 64 -> 32 -> 1)

Ejecución: python generar_figuras_modelo.py
Las curvas con datos reales (ROC, umbral, matrices, SHAP) las genera el notebook 03.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

FIG_DIR = Path(__file__).resolve().parent / "docs" / "figuras_modelo_matematico"
FIG_DIR.mkdir(parents=True, exist_ok=True)

AZUL = "#1565c0"
VERDE = "#2e7d32"
ROJO = "#c62828"
GRIS = "#555555"


def _caja(ax, x, y, texto, fc="#eef3fb", w=0.24, h=0.11, fs=9, ec="#37474f"):
    caja = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.012", facecolor=fc, edgecolor=ec, linewidth=1.3,
    )
    ax.add_patch(caja)
    ax.text(x, y, texto, ha="center", va="center", fontsize=fs)


def _flecha(ax, x1, y1, x2, y2, etiqueta=None, dx=0.012):
    ax.annotate(
        "", (x2, y2), (x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=GRIS, lw=1.2, shrinkA=2, shrinkB=2),
    )
    if etiqueta:
        ax.text((x1 + x2) / 2 + dx, (y1 + y2) / 2, etiqueta, fontsize=8, color="#333333")


def fig_sigmoide_bce():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    # a) Sigmoide
    z = np.linspace(-8, 8, 400)
    s = 1 / (1 + np.exp(-z))
    ax1.plot(z, s, lw=2.6, color=VERDE, zorder=3)
    ax1.axvline(0, color="gray", lw=0.9, ls="--")
    ax1.axhline(0.5, color="gray", lw=0.9, ls="--")
    ax1.fill_between(z, s, 0, where=(s < 0.5), alpha=0.12, color=VERDE)
    ax1.fill_between(z, s, 1, where=(s >= 0.5), alpha=0.12, color=ROJO)
    ax1.text(3.6, 0.58, r"$\hat{y}=1$ (venenoso)", color=ROJO, fontsize=10)
    ax1.text(-7.6, 0.30, r"$\hat{y}=0$ (comestible)", color=VERDE, fontsize=10)
    ax1.text(0.25, 0.515, r"$z=0$", fontsize=8, color="gray")
    ax1.set_xlabel(r"predictor lineal  $z = w_0 + \mathbf{w}^T\mathbf{x}$")
    ax1.set_ylabel(r"$P(Y=1 \mid \mathbf{x}) = \sigma(z)$")
    ax1.set_title(r"a) Función de hipótesis: sigmoide $\sigma(z) = 1/(1+e^{-z})$", fontsize=11)
    ax1.set_ylim(0, 1.02)
    ax1.grid(alpha=0.25)

    # b) BCE
    p = np.linspace(0.002, 0.998, 400)
    ax2.plot(p, -np.log(p), lw=2.3, color=ROJO, label=r"$y=1:\; -\ln(\hat{p})$")
    ax2.plot(p, -np.log(1 - p), lw=2.3, ls="--", color=AZUL, label=r"$y=0:\; -\ln(1-\hat{p})$")
    ax2.set_ylim(0, 6)
    ax2.set_xlabel(r"probabilidad predicha  $\hat{p} = \sigma(\mathbf{w}^T\mathbf{x})$")
    ax2.set_ylabel("pérdida por instancia")
    ax2.set_title("b) Pérdida Binary Cross-Entropy (penaliza el error fatal)", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.25)
    ax2.annotate("error fatal:\n$p\\to 0$ con $y=1$", xy=(0.06, 2.8), xytext=(0.28, 4.2),
                 fontsize=9, color=ROJO,
                 arrowprops=dict(arrowstyle="->", color=ROJO, lw=1.1))

    fig.tight_layout()
    fig.savefig(FIG_DIR / "esquema_sigmoide_bce.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_arbol_bosque():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for ax in (ax1, ax2):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    # a) Un árbol CART
    ax1.set_title("a) Un árbol de decisión CART (divide maximizando la ganancia de Gini)", fontsize=11)
    _caja(ax1, 0.25, 0.88, "¿gill-color = negro?", fc="#fff3e0", w=0.30)
    _caja(ax1, 0.16, 0.58, "¿cap-diameter ≤ 8 cm?", fc="#fff3e0", w=0.24)
    _caja(ax1, 0.44, 0.58, "Venenoso (p)", fc="#ffcdd2", ec=ROJO, w=0.19)
    _caja(ax1, 0.085, 0.28, "Comestible (e)", fc="#c8e6c9", ec=VERDE, w=0.16, fs=8.5)
    _caja(ax1, 0.27, 0.28, "Venenoso (p)", fc="#ffcdd2", ec=ROJO, w=0.16, fs=8.5)
    _flecha(ax1, 0.17, 0.84, 0.165, 0.65, "no", dx=-0.045)
    _flecha(ax1, 0.33, 0.84, 0.41, 0.65, "sí")
    _flecha(ax1, 0.115, 0.525, 0.095, 0.345, "sí", dx=-0.045)
    _flecha(ax1, 0.205, 0.525, 0.25, 0.345, "no")
    ax1.text(0.22, 0.10, r"$I_G(t) = 1 - \sum_k p_k^2(t)$", ha="center", fontsize=10)

    # b) Bagging (flujo vertical: dataset → árboles → agregación → dictamen)
    ax2.set_title("b) Random Forest: bagging de B árboles + agregación", fontsize=11)
    _caja(ax2, 0.69, 0.87, "muestreo bootstrap con reemplazo", fc="#f5f5f5", w=0.34, h=0.10, fs=8.5)
    for x, nombre in [(0.46, "Árbol $T_1$"), (0.69, "Árbol $T_2$"), (0.92, "Árbol $T_B$")]:
        _caja(ax2, x, 0.62, nombre, fc="#e3f2fd", w=0.14, h=0.10)
    ax2.text(1.005, 0.62, "⋯", fontsize=14, ha="center")
    for x in (0.46, 0.69, 0.92):
        _flecha(ax2, 0.69 + (x - 0.69) * 0.28, 0.815, x, 0.675)
        _flecha(ax2, x, 0.565, 0.69 + (x - 0.69) * 0.28, 0.405)
    _caja(ax2, 0.69, 0.34, r"$\hat{P}(Y{=}1|\mathbf{x}) = \frac{1}{B}\sum_{b=1}^{B} P_b(Y{=}1|\mathbf{x})$",
          fc="#e8f5e9", w=0.40, h=0.13, fs=10)
    _caja(ax2, 0.69, 0.10, "votación de mayoría / promedio → dictamen p / e", fc="#ffffff", w=0.42, h=0.10, fs=8.5)
    _flecha(ax2, 0.69, 0.27, 0.69, 0.155)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "esquema_arbol_bosque.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_boosting():
    fig, ax = plt.subplots(figsize=(11.5, 4.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("XGBoost: ensamble secuencial aditivo — cada árbol corrige los errores acumulados", fontsize=12)

    _caja(ax, 0.10, 0.55, "$F_0 = 0$", fc="#f5f5f5", w=0.14)
    xs = [0.30, 0.50, 0.70]
    for i, x in enumerate(xs, start=1):
        _caja(ax, x, 0.55, f"$f_{i}(\\mathbf{{x}})$\nÁrbol CART", fc="#fff3e0", w=0.16, h=0.16)
        _flecha(ax, x - 0.12, 0.55, x - 0.085, 0.55)
    _caja(ax, 0.90, 0.55, r"$\hat{y} = \sigma\!\left(\sum_k f_k\right)$", fc="#e8f5e9", w=0.15, h=0.14)
    _flecha(ax, 0.78, 0.55, 0.825, 0.55)

    ax.text(0.30, 0.33, r"minimiza $\mathcal{L}^{(1)}$", ha="center", fontsize=9, color=GRIS)
    ax.text(0.50, 0.33, r"minimiza $\mathcal{L}^{(2)}$", ha="center", fontsize=9, color=GRIS)
    ax.text(0.70, 0.33, r"minimiza $\mathcal{L}^{(3)}$", ha="center", fontsize=9, color=GRIS)
    ax.text(0.50, 0.14,
            r"$\mathcal{L}^{(t)} = \sum_i l\!\left(y_i, \hat{y}_i^{(t-1)} + f_t(\mathbf{x}_i)\right) + \Omega(f_t)"
            r"\qquad \Omega(f_t) = \gamma T + \frac{1}{2}\lambda \sum_j w_j^2$",
            ha="center", fontsize=11)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "esquema_boosting.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_mlp():
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("MLPClassifier: propagación hacia adelante 128 → 64 → 32 → 1 (ReLU + sigmoide)", fontsize=12)

    capas = {
        "entrada": {"x": 0.10, "n": 5, "color": "#e3f2fd", "etiqueta": "entrada\n128 features\n(preprocesadas)"},
        "oculta1": {"x": 0.37, "n": 4, "color": "#fff3e0", "etiqueta": "capa oculta 1\n64 neuronas\nReLU"},
        "oculta2": {"x": 0.64, "n": 3, "color": "#fff3e0", "etiqueta": "capa oculta 2\n32 neuronas\nReLU"},
        "salida": {"x": 0.90, "n": 1, "color": "#e8f5e9", "etiqueta": "salida\n1 neurona\n$\\sigma(z)$"},
    }

    posiciones = {}
    for nombre, cfg in capas.items():
        n = cfg["n"]
        ys = np.linspace(0.72, 0.24, n)
        posiciones[nombre] = [(cfg["x"], y) for y in ys]
        for x, y in posiciones[nombre]:
            circ = plt.Circle((x, y), 0.028, facecolor=cfg["color"], edgecolor="#37474f", lw=1.2, zorder=3)
            ax.add_patch(circ)
        if n > 1:
            ax.text(cfg["x"], ys[-1] - 0.055, "⋮", ha="center", fontsize=13)
        ax.text(cfg["x"], 0.065, cfg["etiqueta"], ha="center", fontsize=9)

    claves = list(capas)
    for a, b in zip(claves, claves[1:]):
        for (xa, ya) in posiciones[a]:
            for (xb, yb) in posiciones[b]:
                ax.plot([xa + 0.028, xb - 0.028], [ya, yb], color="#90a4ae", lw=0.5, alpha=0.45, zorder=1)

    ax.text(0.5, 0.955, r"$\mathbf{z}^{[l]} = \mathbf{W}^{[l]}\mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}"
                        r"\qquad \mathbf{a}^{[l]} = g(\mathbf{z}^{[l]}) \qquad g = \mathrm{ReLU}(z)=\max(0,z)$",
            ha="center", fontsize=11)
    ax.text(0.955, 0.485, r"$P(Y{=}1)$", fontsize=8, ha="left")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "esquema_mlp.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_sigmoide_bce()
    fig_arbol_bosque()
    fig_boosting()
    fig_mlp()
    print(f"OK - esquemas generados en {FIG_DIR}")
    for f in sorted(FIG_DIR.glob("esquema_*.png")):
        print(f"  • {f.name}")
