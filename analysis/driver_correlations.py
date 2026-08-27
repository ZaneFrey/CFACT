"""Autocorrelation-analysis driver."""

from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from pathlib import Path

import numpy as np

from analysis.driver_common import (
    artifact_from_figure,
    driver_parser,
    load_data,
    load_driver_config,
    resolve_flags,
)
from analysis.models import PlotArtifact
from analysis.statistics import compute_autocorrelation
from tools.common import get_variable_time_axis
from tools.plotting import plot_autocorrelation
from tools.series import collect_height_series
from tools.style import apply_style

PLOT_AUTOCORRELATION = False
SAVE_FIGURES = True

AUTOCORRELATION_COMPONENTS = ("u", "v", "w", "tc")
MAX_LAG_SECONDS = 900


def run(config_path: str | Path | None = None, flag_overrides: dict[str, bool] | None = None) -> list[PlotArtifact]:
    flags = resolve_flags(
        {
            "plot_autocorrelation": PLOT_AUTOCORRELATION,
            "save_figures": SAVE_FIGURES,
        },
        flag_overrides,
    )
    if not flags["plot_autocorrelation"]:
        return []
    config = load_driver_config(config_path)
    data = load_data(config, list(AUTOCORRELATION_COMPONENTS))
    _, time_local, _, _ = get_variable_time_axis(data)
    apply_style(config.figure)
    stats = []
    for prefix in AUTOCORRELATION_COMPONENTS:
        output = []
        lags = np.array([])
        for entry in collect_height_series(data, config.site, prefix):
            lags, values = compute_autocorrelation(entry["data"], entry["meta"], time_local, MAX_LAG_SECONDS)
            output.append({**entry, "data": values})
        stats.append({"varName": prefix, "displayName": prefix, "lagSeconds": lags, "series": output})
    figure = plot_autocorrelation(stats, f"{config.site.upper()}: autocorrelation")
    return [artifact_from_figure(config, "autocorrelation", figure, flags["save_figures"])]


def main() -> None:
    args = driver_parser(__doc__ or "Autocorrelation analysis").parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
