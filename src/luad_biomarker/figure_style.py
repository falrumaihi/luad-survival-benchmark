"""Binding manuscript figure style derived from the supplied ``example`` folder.

The palette, typography, panel labels, and three-format export contract deliberately
match ``example/mpl_style.py``. Study-specific semantic assignments are defined here
once and reused across every LUAD figure.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, findfont, fontManager


NPG = [
    "#E64B35",
    "#4DBBD5",
    "#00A087",
    "#3C5488",
    "#F39B7F",
    "#8491B4",
    "#91D1C2",
    "#DC0000",
    "#7E6148",
    "#B09C85",
]

C_INK = "#1A1A1A"
C_GRID = "#D9D9D9"
C_SURVIVALPFN = "#E64B35"
C_CONFORMAL = "#00A087"
C_CLINICAL = "#8491B4"
C_ELASTIC_NET = "#3C5488"
C_RSF = "#4DBBD5"
C_XGBOOST = "#F39B7F"
C_DEEPSURV = "#B09C85"
C_HIGH_RISK = "#E64B35"
C_LOW_RISK = "#4DBBD5"

MODEL_COLORS = {
    "Clinical Cox": C_CLINICAL,
    "Elastic-net Cox": C_ELASTIC_NET,
    "Random survival forest": C_RSF,
    "XGBoost survival": C_XGBOOST,
    "DeepSurv": C_DEEPSURV,
    "SurvivalPFN": C_SURVIVALPFN,
    "Conformal SurvivalPFN": C_CONFORMAL,
}

COHORT_COLORS = {
    "TCGA-LUAD": "#3C5488",
    "GSE72094": "#00A087",
    "GSE68465": "#4DBBD5",
    "GSE31210": "#F39B7F",
}


def _serif_stack() -> str:
    for path in (
        Path("/mnt/c/Windows/Fonts/times.ttf"),
        Path("/mnt/c/Windows/Fonts/timesbd.ttf"),
        Path("/mnt/c/Windows/Fonts/timesi.ttf"),
        Path("/mnt/c/Windows/Fonts/timesbi.ttf"),
    ):
        if path.exists():
            fontManager.addfont(path)
    for name in (
        "Times New Roman",
        "Nimbus Roman No9 L",
        "Liberation Serif",
        "FreeSerif",
        "DejaVu Serif",
    ):
        try:
            if findfont(FontProperties(family=name), fallback_to_default=False):
                return name
        except Exception:
            continue
    return "serif"


FONT = _serif_stack()


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [FONT, "Times New Roman", "DejaVu Serif"],
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "figure.titleweight": "bold",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "mathtext.fontset": "stix",
            "mathtext.default": "bf",
            "axes.edgecolor": C_INK,
            "axes.labelcolor": C_INK,
            "text.color": C_INK,
            "xtick.color": C_INK,
            "ytick.color": C_INK,
            "axes.linewidth": 1.1,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": C_GRID,
            "grid.linewidth": 0.6,
            "grid.alpha": 0.8,
            "axes.axisbelow": True,
            "axes.prop_cycle": mpl.cycler(color=NPG),
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 1.1,
            "ytick.major.width": 1.1,
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "legend.frameon": False,
            "legend.handlelength": 1.4,
            "legend.columnspacing": 1.1,
            "legend.handletextpad": 0.5,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def panel_label(ax, letter: str, dx: float = -0.16, dy: float = 1.06, size: float = 13) -> None:
    ax.text(
        dx,
        dy,
        letter,
        transform=ax.transAxes,
        fontsize=size,
        fontweight="bold",
        va="top",
        ha="left",
        color=C_INK,
        family="serif",
    )


def bold_all(ax) -> None:
    for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        label.set_fontweight("bold")
    legend = ax.get_legend()
    if legend:
        for text in legend.get_texts():
            text.set_fontweight("bold")


def save(fig, output_dir: str | Path, name: str) -> list[dict[str, str | int]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for ax in fig.get_axes():
        bold_all(ax)

    records: list[dict[str, str | int]] = []
    for extension in ("svg", "png", "pdf"):
        path = output_dir / f"{name}.{extension}"
        fig.savefig(path, format=extension)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        records.append(
            {
                "path": path.as_posix(),
                "format": extension,
                "bytes": path.stat().st_size,
                "sha256": digest,
            }
        )
    plt.close(fig)
    return records
