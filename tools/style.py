"""Matplotlib styling shared by all drivers."""

from __future__ import annotations

import matplotlib as mpl

from analysis.config import FigureConfig


def apply_style(config: FigureConfig) -> None:
    mpl.rcParams.update(
        {
            "font.family": config.font_family,
            "axes.titlesize": config.title_size,
            "axes.labelsize": config.label_size,
            "xtick.labelsize": config.tick_size,
            "ytick.labelsize": config.tick_size,
            "legend.fontsize": config.legend_size,
            "lines.linewidth": config.line_width,
            "axes.grid": config.grid,
        }
    )
