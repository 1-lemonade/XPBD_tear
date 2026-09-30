"""Shared visual defaults for static diagnostics and animation frames."""

from __future__ import annotations

PALETTE = {
    "background": "#F4F7FB",
    "panel": "#FFFFFF",
    "text": "#18263A",
    "muted": "#64748B",
    "grid": "#D8E1EC",
    "blue": "#2563EB",
    "teal": "#0F766E",
    "orange": "#D97706",
    "purple": "#7C3AED",
    "red": "#DC2626",
    "navy": "#19324F",
}

STRAIN_CMAP = "magma"
FIGURE_SIZE = (9.0, 6.0)
FIGURE_DPI = 100
FONT_SIZE = 10
TITLE_SIZE = 13


def apply_plot_style(plt) -> None:
    """Apply the same typography, background, and grid to every plot path."""
    plt.rcParams.update(
        {
            "figure.facecolor": PALETTE["background"],
            "axes.facecolor": PALETTE["panel"],
            "axes.edgecolor": PALETTE["grid"],
            "axes.labelcolor": PALETTE["text"],
            "axes.titlecolor": PALETTE["text"],
            "axes.titlesize": TITLE_SIZE,
            "axes.labelsize": FONT_SIZE,
            "font.size": FONT_SIZE,
            "xtick.color": PALETTE["muted"],
            "ytick.color": PALETTE["muted"],
            "text.color": PALETTE["text"],
            "legend.frameon": True,
            "legend.facecolor": PALETTE["panel"],
            "legend.edgecolor": PALETTE["grid"],
            "savefig.facecolor": PALETTE["background"],
        }
    )


def style_axes(ax) -> None:
    ax.grid(True, color=PALETTE["grid"], linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(PALETTE["grid"])
