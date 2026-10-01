# -*- coding: utf-8 -*-
"""Exporta docs/formulas_matematicas_modelos.md a PDF con ecuaciones renderizadas
(KaTeX auto-render) y las figuras embebidas.
Salida: docs/formulas_matematicas_modelos.pdf

Uso: python exportar_formulas_pdf.py   (requiere Microsoft Edge en la ruta estándar)
"""
import re
import subprocess
import tempfile
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent
MD_PATH = ROOT / "docs" / "formulas_matematicas_modelos.md"
PDF_PATH = ROOT / "docs" / "formulas_matematicas_modelos.pdf"
EDGE_CANDIDATES = [
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
]

texto = MD_PATH.read_text(encoding="utf-8")

# Proteger las ecuaciones de la conversión markdown (los _ y ^ se perderían)
formulas = []


def _guardar(m):
    formulas.append(m.group(0))
    return f"MATHPLACEHOLDER{len(formulas) - 1}ENDMATH"


patron = re.compile(r"\$\$.*?\$\$|\$[^$\n]+\$", re.DOTALL)
texto_protegido = patron.sub(_guardar, texto)

html_cuerpo = markdown.markdown(
    texto_protegido,
    extensions=["tables", "fenced_code", "sane_lists", "attr_list"],
)
for i, f in enumerate(formulas):
    # KaTeX auto-render procesa los pares $...$ / $$...$$ de forma determinista.
    html_cuerpo = html_cuerpo.replace(f"MATHPLACEHOLDER{i}ENDMATH", f)

plantilla = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>Fórmulas Matemáticas de los Modelos de Machine Learning</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"
  onload="renderMathInElement(document.body, {delimiters: [{left:'$$',right:'$$',display:true},{left:'$',right:'$',display:false}], throwOnError:false});"></script>
<style>
  body { font-family: "Segoe UI", Arial, sans-serif; margin: 36px 44px; color: #1a1a1a;
         font-size: 13.5px; line-height: 1.55; max-width: 860px; }
  h1 { color: #14385c; font-size: 24px; border-bottom: 3px solid #14385c; padding-bottom: 8px; }
  h2 { color: #1f5382; font-size: 19px; border-bottom: 1px solid #c9d7e4; padding-bottom: 5px; margin-top: 28px; }
  h3 { color: #2a6496; font-size: 15.5px; margin-top: 20px; }
  table { border-collapse: collapse; margin: 14px 0; font-size: 12.5px; }
  th, td { border: 1px solid #b9c6d2; padding: 6px 12px; text-align: left; }
  th { background: #eef3f8; }
  tr:nth-child(even) td { background: #f7fafc; }
  code { background: #f0f2f5; padding: 1px 5px; border-radius: 3px; font-size: 12px; }
  img { max-width: 100%; height: auto; display: block; margin: 12px auto; border: 1px solid #dfe6ec;
        border-radius: 4px; padding: 4px; }
  .katex-display { margin: 14px 0 !important; }
  blockquote { border-left: 4px solid #2a6496; margin-left: 0; padding-left: 14px; color: #444; }
  hr { border: none; border-top: 1px solid #c9d7e4; margin: 22px 0; }
  @page { size: A4; margin: 14mm; }
</style>
</head>
<body>
__CUERPO__
</body>
</html>
"""

html_final = plantilla.replace("__CUERPO__", html_cuerpo)

with tempfile.TemporaryDirectory(dir=ROOT / "docs") as tmp:
    html_path = Path(tmp) / "formulas.html"
    html_path.write_text(html_final, encoding="utf-8")

    edge = next((e for e in EDGE_CANDIDATES if e.exists()), None)
    if edge is None:
        raise SystemExit("No se encontró Microsoft Edge para imprimir a PDF.")

    resultado = subprocess.run(
        [
            str(edge),
            "--headless",
            "--disable-gpu",
            "--no-first-run",
            "--no-pdf-header-footer",
            "--virtual-time-budget=30000",
            f"--print-to-pdf={PDF_PATH}",
            html_path.resolve().as_uri(),
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    print("edge exit:", resultado.returncode)

tamaño = PDF_PATH.stat().st_size / 1024
print(f"OK - PDF generado: {PDF_PATH} ({tamaño:.0f} KB)")
