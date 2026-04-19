"""
Shared plotting setup and helpers.

Import from any script that produces figures:

    from src.plotting import setup_plot_style, save_fig, COL_PRIMARY, COL_ACCENT
    setup_plot_style()
    ...
    save_fig(fig, "01_target_distribution", FIGDIR)

Goal: one place that owns plot style. Change colors / fonts / dpi here and
every plot in the project updates consistently.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


# color palette
COL_PRIMARY = "#2E5C8A"      # deep blue — main fill
COL_ACCENT = "#C44E52"       # red — significant / warning / accent line
COL_MUTED = "#8FA8C4"        # light blue — secondary / non-significant
COL_HIGHLIGHT = "#E8A33D"    # orange — highlight box / median line
COL_GOOD = "#55A868"         # green — train accuracy / positive baseline
COL_GRID = "#E8E8E8"


def setup_plot_style() -> None:
    """Apply project-wide matplotlib + seaborn style. Call once at top of script."""
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
        "font.family": "DejaVu Sans",  # handles unicode subscripts cleanly
    })


def save_fig(fig, name: str, figdir: Path) -> None:
    """
    Save a figure with consistent settings + a confirmation print.
    Closes the figure after saving.
    """
    figdir.mkdir(parents=True, exist_ok=True)
    path = figdir / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [saved] {name}.png")


def label_bars(ax, bars, values, fmt: str = "{:.0f}",
               offset: float = 3, fontsize: int = 8, color: str = "black") -> None:
    """Annotate vertical bars with values above them."""
    for bar, val in zip(bars, values):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + offset,
            fmt.format(val),
            ha="center", va="bottom", fontsize=fontsize, color=color,
        )


def label_hbars(ax, bars, values, fmt: str = "{:.0f}",
                offset_frac: float = 0.01, fontsize: int = 8, color: str = "black") -> None:
    """Annotate horizontal bars with values to their right."""
    xmax = ax.get_xlim()[1]
    offset = xmax * offset_frac
    for bar, val in zip(bars, values):
        w = bar.get_width()
        ax.text(
            w + offset,
            bar.get_y() + bar.get_height() / 2,
            fmt.format(val),
            ha="left", va="center", fontsize=fontsize, color=color,
        )
