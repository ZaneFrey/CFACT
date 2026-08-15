"""Stable public result models used by all drivers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from matplotlib.figure import Figure


@dataclass(frozen=True, slots=True)
class PlotArtifact:
    """A generated plot and its optional on-disk representation."""

    name: str
    figure: Figure
    saved_path: Path | None = None
