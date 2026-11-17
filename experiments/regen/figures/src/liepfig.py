r"""Shared geometry, palette and I/O for the LIEP manuscript figures.

Figures are authored in Plotly and exported to SVG (editable) and PDF (for
\includegraphics). Everything that is a *number* -- a fitted boundary, a median,
a failure rate -- is read from figures/data/, which scripts/export_figure_data.jl
in PhasorNetworks.jl generates from the result CSVs. Nothing here recomputes an
experiment and no figure contains a hand-typed result.

GEOMETRY. kaleido renders through Chromium, so a Plotly `width` is in CSS pixels
at 96/inch, while a PDF point is 1/72 inch. The exported PDF is therefore
0.75 pt per px -- measured, not assumed: a width=371 figure writes a 277.92 pt
page, and 277.92/371 = 0.75 exactly.

The destination is one-column `sn-jnl`, whose `text={31pc,194.25mm}` gives
\textwidth = 31 pc = 372 TeX pt. TeX points are 1/72.27 inch, so that is
370.6 PDF pt. Authoring at FULL_W px and placing with
`\includegraphics[width=\textwidth]` is then a scale factor of exactly 1.000,
which is what makes a font declared at 8 pt actually print at 8 pt.

Use pt() for every size that should mean points: fonts, line widths, margins.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

# ---------------------------------------------------------------- geometry
PT_PER_PX = 0.75                     # measured; see module docstring
PX_PER_PT = 1.0 / PT_PER_PX          # 4/3

TEXTWIDTH_PT = 370.6                 # \textwidth = 31pc = 372 TeX pt
FULL_W = TEXTWIDTH_PT * PX_PER_PT    # 494.13 px
HALF_W = FULL_W / 2


def pt(x: float) -> float:
    """Convert a size in PDF points to the Plotly pixels that render it."""
    return x * PX_PER_PT


# Body is 10 bp on 12 bp. Captions and figure text sit below it.
FS_TICK = pt(7.0)
FS_LABEL = pt(8.0)
FS_PANEL = pt(8.5)      # the bold (a)/(b)/(c) letters
FS_ANNOT = pt(7.0)

# Figure text is set in a grotesque sans against the serif body.
#
# Liberation Sans, NOT Nimbus Sans, and the reason is font embedding. kaleido
# renders through Chromium, which re-emits CFF/OpenType outlines as **Type 3**
# fonts -- preflight tools flag them and publishers routinely reject them.
# Nimbus Sans ships as .otf and hits exactly that path; Liberation Sans ships as
# .ttf and comes through as a proper CID TrueType subset. Both are Helvetica-class
# neo-grotesques, so the change is invisible on the page.
#
# Downstream repair does not work: pdftocairo mis-scales the Type 3 glyph matrix
# and renders subscripts at display size, ghostscript passes Type 3 through, and
# mutool refuses outright. Getting it right at the source is the only fix.
#
# The same trap reopens through glyph fallback: any character Liberation Sans
# lacks -- script capitals like U+1D4A9, for instance -- sends Chromium to an OTF
# math font and a Type 3 subset comes back. Prefer <i>N</i> to a math-alphanumeric
# codepoint, and run `pdffonts figures/*.pdf | grep 'Type 3'` after editing text.
FONT = "Liberation Sans, Arial, Helvetica, sans-serif"

# ---------------------------------------------------------------- palette
# Identical to the \definecolor block at sn-article.tex:64-70, so a figure and
# the §2 TikZ diagrams cannot drift apart.
HOLO = "rgb(30,110,190)"        # the estimator working
ANTIHOLO = "rgb(200,80,30)"     # boundaries, accumulated update
PROBE = "rgb(40,140,90)"        # the probe / global signal
BASIN = "rgb(90,150,200)"
BAD = "rgb(200,120,120)"        # failure
GRAY = "rgb(110,110,110)"       # axes, de-emphasis
HEBB = "rgb(120,60,160)"        # the local/Hebbian factor


def rgba(c: str, a: float) -> str:
    """Take a `rgb(r,g,b)` string to `rgba(r,g,b,a)` for fills and bands."""
    return c.replace("rgb(", "rgba(").replace(")", f",{a})")


# ---------------------------------------------------------------- paths / io
FIGDIR = Path(__file__).resolve().parent.parent
DATA = FIGDIR / "data"


def dat(name: str) -> pd.DataFrame:
    """Read a .dat table written by export_figure_data.jl (`#` lines are provenance)."""
    return pd.read_csv(DATA / name, sep=r"\s+", comment="#")


def macros(fig: str) -> dict[str, str]:
    """Read figN_macros.tex into {name: value} so annotations quote generated numbers."""
    txt = (DATA / f"{fig}_macros.tex").read_text()
    return dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*?)\}\s*$", txt, re.M))


# ---------------------------------------------------------------- theme
def layout(height_pt: float, **kw) -> dict:
    """Base layout. `height_pt` is in points; width is always the full text block."""
    base = dict(
        width=FULL_W,
        height=height_pt * PX_PER_PT,
        font=dict(family=FONT, size=FS_TICK, color="black"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
    )
    base.update(kw)
    return base


AXIS = dict(
    showline=True,
    linecolor=GRAY,
    linewidth=pt(0.5),
    ticks="outside",
    ticklen=pt(2.5),
    tickwidth=pt(0.5),
    tickcolor=GRAY,
    tickfont=dict(size=FS_TICK),
    title=dict(font=dict(size=FS_LABEL)),
    zeroline=False,
    showgrid=False,
    automargin=False,
)


def panel_label(fig: go.Figure, x: float, y: float, letter: str, text: str = "") -> None:
    """A bold (a) in paper coordinates, matching the §2 TikZ idiom."""
    fig.add_annotation(
        x=x, y=y, xref="paper", yref="paper",
        text=f"<b>({letter})</b>" + (f" {text}" if text else ""),
        showarrow=False, xanchor="left", yanchor="bottom",
        font=dict(size=FS_PANEL, color="black"),
    )


def vstack(heights: list[float], gaps: list[float]) -> tuple[list[list[float]], float]:
    """Lay out rows from a budget in points.

    `heights` are row heights top to bottom, `gaps` the space between them, both
    in points. Returns each row's [lo, hi] y-domain as a fraction of the plot
    area, plus the total plot-area height in points. Plotly domains exclude the
    figure margins, so add those separately when sizing the figure.
    """
    total = sum(heights) + sum(gaps)
    doms, y = [], total
    for i, h in enumerate(heights):
        doms.append([(y - h) / total, y / total])
        y -= h + (gaps[i] if i < len(gaps) else 0.0)
    return doms, total


def hsplit(n: int, gap: float = 0.03) -> list[list[float]]:
    """n equal columns across the plot area, separated by `gap` (paper fraction)."""
    w = (1.0 - gap * (n - 1)) / n
    return [[i * (w + gap), i * (w + gap) + w] for i in range(n)]


def save(fig: go.Figure, name: str) -> None:
    """Write the editable SVG and the PDF the manuscript includes."""
    svg = FIGDIR / f"{name}.svg"
    pdf = FIGDIR / f"{name}.pdf"
    fig.write_image(svg)
    fig.write_image(pdf)
    for out in (svg, pdf):
        print(f"  wrote {out.relative_to(FIGDIR.parent)}")
