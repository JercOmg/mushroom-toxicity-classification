# -*- coding: utf-8 -*-
"""Identidad visual de ESPORA: paleta, tipografía, estilos de Streamlit y matplotlib, y
componentes HTML (dictamen, espectro de probabilidad, ficha de rasgos).

Concepto: la esporada — la huella radial que dejan las láminas de un hongo sobre papel.
Fondo «humus», texto «tiza» y tres señales con significado fijo en toda la app:
liquen = comestible, amanita = venenoso, ocre = incierto.
"""
import math
from html import escape
from pathlib import Path

import matplotlib as mpl
from matplotlib import font_manager

# ------------------------------------------------------------------ paleta
HUMUS = "#0D100E"        # fondo
SUPERFICIE = "#151A17"   # tarjetas
ELEVADO = "#1C231F"      # controles
LINEA = "#2A332E"        # bordes y rejillas
TIZA = "#ECE8DD"         # texto principal
MUSGO = "#8D978F"        # texto secundario

LIQUEN = "#B8D46A"       # comestible / marca
AMANITA = "#F2552C"      # venenoso
OCRE = "#E3A83B"         # incierto

PIZARRA = "#9FB4C7"      # series neutras
ARCILLA = "#D98E73"
PERGAMINO = "#C9B79C"

COLOR_ESTADO = {"comestible": LIQUEN, "venenoso": AMANITA, "incierto": OCRE}
COLORES_MODELOS = {
    "Logistic Regression": PIZARRA,
    "Random Forest": LIQUEN,
    "XGBoost": ARCILLA,
    "MLP Neural Network": PERGAMINO,
}

ASSETS = Path(__file__).absolute().parent / "assets"


# ------------------------------------------------------------------ marca: esporada
def esporada_svg(color: str = TIZA, tamano: int = 64, acento: str = LIQUEN, laminas: int = 40,
                 fondo: str = "none") -> str:
    """Isotipo: el envés de un sombrero visto desde abajo. Láminas largas desde el pie,
    laminillas cortas intercaladas y el corte del pie en el centro."""
    c, r = 32, 29
    trazos = []
    for i in range(laminas):
        a = 2 * math.pi * i / laminas - math.pi / 2
        r0 = 7.5 if i % 2 == 0 else 16.5
        x0, y0 = c + r0 * math.cos(a), c + r0 * math.sin(a)
        x1, y1 = c + (r - 2.6) * math.cos(a), c + (r - 2.6) * math.sin(a)
        ancho = 1.45 if i % 2 == 0 else 1.05
        trazos.append(f'<line x1="{x0:.2f}" y1="{y0:.2f}" x2="{x1:.2f}" y2="{y1:.2f}" '
                      f'stroke="{color}" stroke-width="{ancho}" stroke-linecap="round"/>')
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" width="{tamano}" height="{tamano}" '
        f'role="img" aria-label="ESPORA">'
        f'<rect width="64" height="64" rx="14" fill="{fondo}"/>'
        f'<circle cx="{c}" cy="{c}" r="{r}" fill="none" stroke="{color}" stroke-width="1.6"/>'
        + "".join(trazos)
        + f'<circle cx="{c}" cy="{c}" r="4.6" fill="{acento}"/></svg>'
    )


def logotipo_svg(color: str = TIZA, acento: str = LIQUEN) -> str:
    """Isotipo + palabra. El texto usa Bricolage Grotesque con alternativas del sistema."""
    marca = esporada_svg(color, 64, acento).replace('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" '
                                                    'width="64" height="64" role="img" aria-label="ESPORA">', "")
    marca = marca.replace("</svg>", "")
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 64" width="300" height="64" '
        'role="img" aria-label="ESPORA — clasificador de toxicidad micológica">'
        f'<g>{marca}</g>'
        f'<text x="80" y="38" font-family="Bricolage Grotesque, Arial Black, Helvetica, sans-serif" '
        f'font-size="31" font-weight="800" letter-spacing="3" fill="{color}">ESPORA</text>'
        f'<text x="81" y="55" font-family="IBM Plex Mono, Consolas, monospace" font-size="9.2" '
        f'letter-spacing="1.6" fill="{acento}">TOXICIDAD MICOLÓGICA</text></svg>'
    )


def escribir_logos() -> dict:
    """Genera los archivos de marca en app/assets (SVG + PNG para el favicon)."""
    ASSETS.mkdir(parents=True, exist_ok=True)
    archivos = {
        "isotipo": ASSETS / "espora-isotipo.svg",
        "isotipo_claro": ASSETS / "espora-isotipo-claro.svg",
        "logotipo": ASSETS / "espora-logotipo.svg",
        "logotipo_claro": ASSETS / "espora-logotipo-claro.svg",
        "favicon": ASSETS / "espora-favicon.png",
    }
    archivos["isotipo"].write_text(esporada_svg(TIZA, 64, LIQUEN, fondo=HUMUS), encoding="utf-8")
    archivos["isotipo_claro"].write_text(esporada_svg(HUMUS, 64, "#5E7A1F"), encoding="utf-8")
    archivos["logotipo"].write_text(logotipo_svg(TIZA, LIQUEN), encoding="utf-8")
    archivos["logotipo_claro"].write_text(logotipo_svg(HUMUS, "#5E7A1F"), encoding="utf-8")
    if not archivos["favicon"].exists():
        _favicon_png(archivos["favicon"])
    return archivos


def _favicon_png(destino: Path) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(1.28, 1.28), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 64)
    ax.set_ylim(64, 0)
    ax.axis("off")
    fig.patch.set_alpha(0)
    ax.add_patch(mpl.patches.FancyBboxPatch((0, 0), 64, 64, boxstyle="round,pad=0,rounding_size=14",
                                            color=HUMUS))
    ax.add_patch(mpl.patches.Circle((32, 32), 29, fill=False, color=TIZA, lw=2.2))
    for i in range(28):
        a = 2 * math.pi * i / 28
        r0 = 7.5 if i % 2 == 0 else 17
        ax.plot([32 + r0 * math.cos(a), 32 + 26.4 * math.cos(a)],
                [32 + r0 * math.sin(a), 32 + 26.4 * math.sin(a)],
                color=TIZA, lw=2.1 if i % 2 == 0 else 1.4, solid_capstyle="round")
    ax.add_patch(mpl.patches.Circle((32, 32), 5.2, color=LIQUEN))
    fig.savefig(destino, transparent=True)
    plt.close(fig)


# ------------------------------------------------------------------ matplotlib
def configurar_matplotlib() -> None:
    disponibles = {f.name for f in font_manager.fontManager.ttflist}
    familia = next((f for f in ["Bricolage Grotesque", "Schibsted Grotesk", "Segoe UI", "Helvetica Neue",
                                "Arial", "DejaVu Sans"] if f in disponibles), "DejaVu Sans")
    mpl.rcParams.update({
        "figure.facecolor": SUPERFICIE,
        "axes.facecolor": SUPERFICIE,
        "savefig.facecolor": SUPERFICIE,
        "axes.edgecolor": LINEA,
        "axes.labelcolor": MUSGO,
        "axes.titlecolor": TIZA,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 12,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.prop_cycle": mpl.cycler(color=[LIQUEN, PIZARRA, ARCILLA, PERGAMINO, OCRE, AMANITA]),
        "grid.color": LINEA,
        "grid.linewidth": 0.8,
        "xtick.color": MUSGO,
        "ytick.color": MUSGO,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "text.color": TIZA,
        "font.family": familia,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "legend.labelcolor": MUSGO,
        "patch.edgecolor": "none",
        "figure.dpi": 110,
    })


def mapa_secuencial(color: str = LIQUEN):
    return mpl.colors.LinearSegmentedColormap.from_list("espora", [ELEVADO, color])


# ------------------------------------------------------------------ CSS
FUENTES = (
    "https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,400..800"
    "&family=IBM+Plex+Mono:wght@400;500;600&family=Schibsted+Grotesk:wght@400..700&display=swap"
)

CSS = f"""
<style>
@import url('{FUENTES}');
:root {{
  --humus:{HUMUS}; --superficie:{SUPERFICIE}; --elevado:{ELEVADO}; --linea:{LINEA};
  --tiza:{TIZA}; --musgo:{MUSGO}; --liquen:{LIQUEN}; --amanita:{AMANITA}; --ocre:{OCRE};
  --display:'Bricolage Grotesque', 'Segoe UI', sans-serif;
  --texto:'Schibsted Grotesk', 'Segoe UI', sans-serif;
  --mono:'IBM Plex Mono', Consolas, monospace;
}}
html, body, .stApp, p, li, label, input, textarea, button, td, th,
[data-testid="stMarkdownContainer"], [data-testid="stCaptionContainer"], [data-baseweb="select"] div {{
  font-family: var(--texto);
}}
.stApp {{
  background:
    radial-gradient(1200px 520px at 100% -10%, rgba(184,212,106,.055), transparent 60%),
    var(--humus);
}}
[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 2.2rem; max-width: 1240px; }}
h1, h2, h3, h4 {{ font-family: var(--display) !important; letter-spacing: -.01em; color: var(--tiza); }}
h4 {{ font-size: 1.05rem !important; font-weight: 700 !important; margin-top: 1.4rem; }}
p, li {{ color: #D4D1C7; }}
[data-testid="stCaptionContainer"] p {{ color: var(--musgo); }}
code {{ font-family: var(--mono) !important; color: var(--liquen) !important; background: var(--elevado) !important; }}
hr {{ border-color: var(--linea) !important; }}

/* ---------- barra lateral */
[data-testid="stSidebar"] {{ background: #0A0C0B; border-right: 1px solid var(--linea); }}
[data-testid="stSidebar"] [data-testid="stRadio"] > label {{ display: none; }}
[data-testid="stSidebar"] [role="radiogroup"] {{ gap: 2px; }}
[data-testid="stSidebar"] [role="radiogroup"] label {{
  padding: 8px 10px; border-radius: 8px; border: 1px solid transparent; width: 100%;
  transition: background .15s, border-color .15s;
}}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {{ background: var(--superficie); }}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked),
[data-testid="stSidebar"] [role="radiogroup"] label[data-selected="true"] {{
  background: var(--superficie); border-color: var(--linea);
}}
[data-testid="stSidebar"] [role="radiogroup"] label[data-selected="true"] p {{ font-weight: 600; }}
[data-testid="stSidebar"] [role="radiogroup"] label:not([data-selected="true"]) p {{ color: var(--musgo); }}
[data-testid="stSidebar"] [role="radiogroup"] label > div > div:not([data-testid]) {{ display: none; }}
[data-testid="stSidebar"] [role="radiogroup"] label p {{ font-size: .92rem; color: var(--tiza); }}

/* ---------- formularios y controles */
[data-testid="stForm"] {{
  background: var(--superficie); border: 1px solid var(--linea); border-radius: 14px; padding: 1.4rem 1.4rem .8rem;
}}
[data-baseweb="select"] > div, [data-baseweb="input"] > div, .stNumberInput input {{
  background: var(--elevado) !important; border-color: var(--linea) !important; border-radius: 9px !important;
}}
.stButton button, [data-testid="stFormSubmitButton"] button, [data-testid="stDownloadButton"] button {{
  font-family: var(--texto); font-weight: 600; border-radius: 10px; border: 1px solid var(--linea);
  background: var(--elevado); color: var(--tiza);
}}
[data-testid="stFormSubmitButton"] button {{
  background: var(--liquen); color: #10140C; border: none; padding: .7rem 1rem; letter-spacing: .02em;
}}
[data-testid="stFormSubmitButton"] button:hover {{ background: #C9E27E; color: #10140C; }}
[data-testid="stExpander"] details {{ border: 1px solid var(--linea); border-radius: 12px; background: var(--superficie); }}
[data-testid="stAlert"] {{ border-radius: 12px; }}

/* ---------- métricas */
[data-testid="stMetric"] {{
  background: var(--superficie); border: 1px solid var(--linea); border-radius: 12px; padding: 14px 16px;
}}
[data-testid="stMetricLabel"] p {{
  font-family: var(--mono); font-size: .72rem; letter-spacing: .08em; text-transform: uppercase; color: var(--musgo);
}}
[data-testid="stMetricValue"] {{ font-family: var(--display); font-weight: 700; color: var(--tiza); }}

/* ---------- gráficos y tablas */
[data-testid="stImage"] img, [data-testid="stPyplot"] img, .stPyplot img {{ border-radius: 12px; }}
[data-testid="stDataFrame"] {{ border: 1px solid var(--linea); border-radius: 12px; overflow: hidden; }}

/* ---------- componentes ESPORA */
.esp-kicker {{ font-family: var(--mono); font-size: .72rem; letter-spacing: .14em; text-transform: uppercase; color: var(--musgo); }}
.esp-kicker b {{ color: var(--liquen); font-weight: 500; }}
.esp-cabecera {{ display: flex; align-items: flex-end; justify-content: space-between; gap: 24px;
  padding-bottom: 18px; margin-bottom: 22px; border-bottom: 1px solid var(--linea); flex-wrap: wrap; }}
.esp-cabecera h1 {{ font-size: clamp(1.9rem, 3.4vw, 2.7rem); font-weight: 800; line-height: 1.02; margin: 8px 0 6px; padding: 0; }}
.esp-cabecera p {{ color: var(--musgo); margin: 0; max-width: 640px; }}
.esp-indice {{ font-family: var(--display); font-weight: 800; font-size: 4.2rem; line-height: .8;
  color: transparent; -webkit-text-stroke: 1px var(--linea); }}
@media (max-width: 1100px) {{ .esp-indice {{ display: none; }} }}
.esp-marca {{ display: flex; align-items: center; gap: 12px; padding: 4px 2px 18px; }}
.esp-marca .nombre {{ font-family: var(--display); font-weight: 800; font-size: 1.45rem; letter-spacing: .12em; color: var(--tiza); line-height: 1; }}
.esp-marca .sub {{ font-family: var(--mono); font-size: .62rem; letter-spacing: .16em; color: var(--liquen); margin-top: 5px; }}
.esp-ficha {{ font-family: var(--mono); font-size: .74rem; color: var(--musgo); border-top: 1px solid var(--linea);
  padding-top: 14px; margin-top: 10px; line-height: 1.9; }}
.esp-ficha span {{ color: var(--tiza); float: right; }}
.esp-seccion {{ font-family: var(--mono); font-size: .72rem; letter-spacing: .14em; text-transform: uppercase;
  color: var(--musgo); margin: 14px 0 6px; display: flex; align-items: center; gap: 10px; }}
[data-testid="stForm"] [data-testid="stElementContainer"]:first-child .esp-seccion {{ margin-top: 0; }}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{ margin-top: 14px; }}
.esp-seccion::after {{ content: ""; flex: 1; height: 1px; background: var(--linea); }}

.esp-veredicto {{ display: grid; grid-template-columns: auto 1fr auto; gap: 26px; align-items: center;
  background: var(--superficie); border: 1px solid var(--linea); border-radius: 18px; padding: 22px 26px;
  position: relative; overflow: hidden; margin-top: 8px; }}
.esp-veredicto::before {{ content: ""; position: absolute; inset: 0 auto 0 0; width: 4px; background: var(--senal); }}
.esp-veredicto .titulo {{ font-family: var(--display); font-weight: 800; font-size: clamp(2rem, 4vw, 3.1rem);
  line-height: 1; color: var(--senal); margin: 6px 0 8px; letter-spacing: -.01em; }}
.esp-veredicto .texto {{ color: #D4D1C7; max-width: 560px; font-size: .95rem; }}
.esp-veredicto .dato {{ text-align: right; }}
.esp-veredicto .num {{ font-family: var(--display); font-weight: 700; font-size: 2.3rem; color: var(--tiza); line-height: 1; }}
@media (max-width: 760px) {{ .esp-veredicto {{ grid-template-columns: 1fr; }} .esp-veredicto .dato {{ text-align: left; }} }}

.esp-espectro {{ margin: 18px 0 6px; }}
.esp-espectro .pista {{ position: relative; height: 14px; border-radius: 7px; overflow: visible; display: flex; }}
.esp-espectro .zona {{ height: 100%; }}
.esp-espectro .marcador {{ position: absolute; top: -7px; width: 3px; height: 28px; background: var(--tiza);
  border-radius: 2px; transform: translateX(-1.5px); box-shadow: 0 0 0 3px var(--humus); }}
.esp-espectro .tau {{ position: absolute; top: -3px; height: 20px; border-left: 1px dashed var(--tiza); opacity: .7; }}
.esp-espectro .reglas {{ position: relative; height: 22px; font-family: var(--mono); font-size: .68rem; color: var(--musgo); margin-top: 8px; }}
.esp-espectro .reglas span {{ position: absolute; transform: translateX(-50%); white-space: nowrap; }}

.esp-rasgos {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(185px, 1fr)); gap: 6px; margin-top: 10px; }}
.esp-rasgo {{ font-size: .8rem; padding: 7px 10px; border-radius: 8px; border: 1px solid var(--linea);
  background: var(--superficie); color: #D4D1C7; display: flex; gap: 8px; align-items: center; }}
.esp-rasgo i {{ width: 8px; height: 8px; border-radius: 50%; flex: none; background: var(--liquen); }}
.esp-rasgo.fuera {{ border-color: rgba(227,168,59,.45); color: var(--tiza); }}
.esp-rasgo.fuera i {{ background: transparent; border: 1.5px solid var(--ocre); }}

.esp-nota {{ border: 1px solid var(--linea); border-left: 3px solid var(--ocre); background: var(--superficie);
  border-radius: 10px; padding: 14px 16px; color: #D4D1C7; font-size: .9rem; margin: 10px 0; }}
.esp-nota b {{ color: var(--tiza); }}
.esp-cifras {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin: 8px 0 4px; }}
@media (max-width: 1100px) {{ .esp-cifras {{ grid-template-columns: repeat(2, 1fr); }} }}
.esp-cifra {{ background: var(--superficie); border: 1px solid var(--linea); border-radius: 12px; padding: 14px 16px; }}
.esp-cifra .k {{ font-family: var(--mono); font-size: .68rem; letter-spacing: .1em; text-transform: uppercase; color: var(--musgo); }}
.esp-cifra .v {{ font-family: var(--display); font-weight: 800; font-size: clamp(1.3rem, 2.4vw, 1.9rem); color: var(--tiza);
  line-height: 1.1; margin-top: 4px; overflow-wrap: anywhere; }}
.esp-cifra .d {{ font-size: .78rem; color: var(--musgo); }}
.esp-cifra.mala .v {{ color: var(--amanita); }} .esp-cifra.buena .v {{ color: var(--liquen); }}
</style>
"""


# ------------------------------------------------------------------ componentes
def marca_lateral() -> str:
    return (f'<div class="esp-marca">{esporada_svg(TIZA, 42, LIQUEN)}'
            f'<div><div class="nombre">ESPORA</div><div class="sub">TOXICIDAD MICOLÓGICA</div></div></div>')


def cabecera(indice: str, titulo: str, bajada: str, kicker: str) -> str:
    return (f'<div class="esp-cabecera"><div><div class="esp-kicker">{kicker}</div>'
            f'<h1>{escape(titulo)}</h1><p>{bajada}</p></div><div class="esp-indice">{indice}</div></div>')


def _tau(html: str) -> str:
    """Evita que text-transform: uppercase convierta τ en Τ."""
    return html.replace("τ", '<span style="text-transform:none">τ</span>')


def seccion(texto: str) -> str:
    return f'<div class="esp-seccion">{_tau(escape(texto))}</div>'


TITULOS_ESTADO = {"comestible": "Comestible", "venenoso": "Venenoso", "incierto": "Incierto"}


def veredicto(estado: str, texto: str, proba: float) -> str:
    color = COLOR_ESTADO[estado]
    return (
        f'<div class="esp-veredicto" style="--senal:{color}">'
        f'{esporada_svg(color, 92, color, laminas=44)}'
        f'<div><div class="esp-kicker">Dictamen del sistema</div>'
        f'<div class="titulo">{TITULOS_ESTADO[estado]}</div><div class="texto">{texto}</div></div>'
        f'<div class="dato"><div class="esp-kicker">P(venenoso)</div><div class="num">{proba:.1%}</div></div></div>'
    )


def espectro(proba: float, tau: float, franja: tuple) -> str:
    a, b = franja
    pct = lambda v: f"{max(0.0, min(1.0, v)) * 100:.2f}%"  # noqa: E731
    zonas = (
        f'<div class="zona" style="width:{pct(a)};background:{LIQUEN};opacity:.85;border-radius:7px 0 0 7px"></div>'
        f'<div class="zona" style="width:{pct(b - a)};background:repeating-linear-gradient(135deg,{OCRE} 0 5px,'
        f'rgba(227,168,59,.45) 5px 10px)"></div>'
        f'<div class="zona" style="flex:1;background:{AMANITA};opacity:.85;border-radius:0 7px 7px 0"></div>'
    )
    reglas = "".join(f'<span style="left:{pct(v)}">{t}</span>' for v, t in
                     [(0, "0"), (a, f"{a:.2f}"), (tau, f"τ* {tau:.2f}"), (b, f"{b:.2f}"), (1, "1")])
    return (
        f'<div class="esp-espectro"><div class="pista">{zonas}'
        f'<div class="tau" style="left:{pct(tau)}"></div>'
        f'<div class="marcador" style="left:{pct(min(max(proba, 0.004), 0.996))}"></div></div>'
        f'<div class="reglas">{reglas}</div>'
        f'<div class="esp-kicker" style="display:flex;gap:18px;flex-wrap:wrap;margin-top:2px">'
        f'<span><b style="color:{LIQUEN}">■</b> comestible</span>'
        f'<span><b style="color:{OCRE}">■</b> sin datos de entrenamiento</span>'
        f'<span><b style="color:{AMANITA}">■</b> venenoso</span>'
        f'<span><b style="color:{TIZA}">▮</b> este espécimen</span></div></div>'
    )


def rasgos(etiquetas: dict, fuera: list) -> str:
    celdas = "".join(
        f'<div class="esp-rasgo{" fuera" if c in fuera else ""}"><i></i>{escape(n)}</div>'
        for c, n in etiquetas.items()
    )
    return f'<div class="esp-rasgos">{celdas}</div>'


def nota(html: str) -> str:
    return f'<div class="esp-nota">{html}</div>'


def cifras(items: list, columnas: int = None) -> str:
    """items: (clave, valor, detalle, tono) con tono '', 'buena' o 'mala'."""
    estilo = f' style="grid-template-columns:repeat({columnas},1fr)"' if columnas else ""
    return f'<div class="esp-cifras"{estilo}>' + "".join(
        f'<div class="esp-cifra {t}"><div class="k">{_tau(k)}</div><div class="v">{v}</div><div class="d">{d}</div></div>'
        for k, v, d, t in items
    ) + "</div>"
