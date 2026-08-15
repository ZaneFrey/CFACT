"""Figure naming and persistence."""

from __future__ import annotations

from pathlib import Path

from matplotlib.figure import Figure

from analysis.models import PlotArtifact


def save_figure(
    name: str,
    figure: Figure,
    output_dir: Path,
    output_format: str,
    dpi: int,
    overwrite: bool,
    enabled: bool,
) -> PlotArtifact:
    """Create a plot artifact and, when requested, save it deterministically."""

    path: Path | None = None
    if enabled:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{name}.{output_format}"
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing figure: {path}")
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        if not path.is_file() or path.stat().st_size == 0:
            raise OSError(f"Figure was not written successfully: {path}")
    return PlotArtifact(name=name, figure=figure, saved_path=path)
