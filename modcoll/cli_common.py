"""Shared helpers of the modcoll command-line tools: argument parsing of
lists, CSV/JSON writing and one figure style."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

# Figure palette (light surface). Categorical slots are used in this fixed
# order and never cycled; the ordinal ramp is one hue, light -> dark, for an
# ordered variable such as the modification fraction.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
CATEGORICAL = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
ORDINAL_BLUE = ("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b")
MARKERS = ("o", "s", "^", "D", "v")


def float_list(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def ordinal_colors(n: int) -> list[str]:
    """``n`` steps of the one-hue ramp, light to dark (n <= 5 distinct)."""
    if n <= 1:
        return [ORDINAL_BLUE[2]]
    idx = np.round(np.linspace(0, len(ORDINAL_BLUE) - 1, n)).astype(int)
    return [ORDINAL_BLUE[i] for i in idx]


def level_styles(n: int) -> list[tuple[str, str]]:
    """(colour, marker) for ``n`` levels of one variable: the categorical
    slots while they last (n <= 4), otherwise the one-hue ramp, which suits
    the ordered variables (pH) that come in more levels. Never cycles a
    colour; markers repeat only beyond five levels."""
    colors = list(CATEGORICAL[:n]) if n <= len(CATEGORICAL) else ordinal_colors(n)
    return [(colors[k], MARKERS[k % len(MARKERS)]) for k in range(n)]


def pyplot():
    """matplotlib.pyplot on the non-interactive backend with the shared style."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "axes.edgecolor": INK_SECONDARY,
        "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "font.size": 9, "axes.titlesize": 9.5,
        "axes.titleweight": "bold", "axes.titlelocation": "left",
        "legend.frameon": False, "lines.linewidth": 1.6, "lines.markersize": 5,
        "figure.dpi": 100, "savefig.dpi": 150,
    })
    return plt


def write_csv(path, rows: list[dict]) -> None:
    """Rows as CSV, columns in first-seen order; floats to 6 significant
    digits so repeated runs give byte-identical files."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols: list[str] = []
    for row in rows:
        for key in row:
            if key not in cols:
                cols.append(key)

    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if not np.isfinite(v) else f"{float(v):.6g}"
        return v

    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow({k: fmt(v) for k, v in row.items()})


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def write_json(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(obj), indent=2, sort_keys=True) + "\n")


def headline(fig, title: str, subtitle: str = "") -> tuple:
    """Left-aligned figure title with an optional secondary line, placed at
    fixed distances from the top edge. Returns the ``rect`` to hand to
    ``fig.tight_layout`` so the axes stay clear of both lines."""
    height = fig.get_figheight()
    fig.text(0.01, 1.0 - 0.10 / height, title, ha="left", va="top", fontsize=10.5,
             fontweight="bold", color=INK)
    if subtitle:
        fig.text(0.01, 1.0 - 0.32 / height, subtitle, ha="left", va="top", fontsize=8.5,
                 color=INK_SECONDARY)
    return (0, 0, 1, 1.0 - (0.58 if subtitle else 0.36) / height)
