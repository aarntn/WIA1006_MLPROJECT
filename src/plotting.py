"""
Shared plotting setup and helpers for Swipe Atlas.

Import from any script that produces figures:

    from src.plotting import setup_plot_style, save_fig, COL_PRIMARY, COL_ACCENT

This is the single source of truth for plot style. Change colors/fonts/dpi
here and every figure in the project updates consistently.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


# Color palette — used across all figures for visual consistency
COL_PRIMARY = "#2E5C8A"    # deep blue — main bars/fills
COL_ACCENT = "#C44E52"     # red — significant results, warning lines
COL_MUTED = "#8FA8C4"      # light blue — secondary/non-significant
COL_HIGHLIGHT = "#E8A33D"  # orange — highlights, median lines
COL_GOOD = "#55A868"       # green — positive baselines, train accuracy
COL_GRID = "#E8E8E8"


def setup_plot_style() -> None:
    """Apply project-wide matplotlib + seaborn style. Call once at top of each script."""
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "font.family": "DejaVu Sans",
    })


def save_fig(fig, name: str, figdir: Path) -> None:
    """Save figure to figdir/{name}.png with consistent settings, then close it."""
    figdir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(figdir / f"{name}.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [saved] {name}.png")


