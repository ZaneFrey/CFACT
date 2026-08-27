"""Quadrant-analysis driver."""

from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from pathlib import Path

from analysis._math import expand_raw_series, moving_mean
from analysis.driver_common import (
    align_height_series,
    artifact_from_figure,
    driver_parser,
    load_data,
    load_driver_config,
    resolve_flags,
)
from analysis.models import PlotArtifact
from tools.common import get_variable_time_axis
from tools.plotting import plot_quadrant_joint_pdf, plot_quadrant_scatter
from tools.series import collect_height_series
from tools.style import apply_style

PLOT_QUADRANT_SCATTER = False
PLOT_QUADRANT_JOINT_PDF = False
SAVE_FIGURES = True


def run(config_path: str | Path | None = None, flag_overrides: dict[str, bool] | None = None) -> list[PlotArtifact]:
    flags = resolve_flags(
        {
            "plot_quadrant_scatter": PLOT_QUADRANT_SCATTER,
            "plot_quadrant_joint_pdf": PLOT_QUADRANT_JOINT_PDF,
            "save_figures": SAVE_FIGURES,
        },
        flag_overrides,
    )
    if not (flags["plot_quadrant_scatter"] or flags["plot_quadrant_joint_pdf"]):
        return []
    config = load_driver_config(config_path)
    data = load_data(config, ["u", "w"])
    _, time_local, _, _ = get_variable_time_axis(data)
    apply_style(config.figure)
    u_series = collect_height_series(data, config.site, "u")
    w_series = collect_height_series(data, config.site, "w")
    u_series, w_series = align_height_series(u_series, w_series)
    u_prime = []
    w_prime = []
    for u_entry, w_entry in zip(u_series, w_series):
        u_raw, _, rate, _ = expand_raw_series(u_entry["data"], u_entry["meta"], time_local)
        w_raw, _, _, _ = expand_raw_series(w_entry["data"], w_entry["meta"], time_local)
        n = min(u_raw.size, w_raw.size)
        window = max(1, int(round(config.averaging_period_seconds * rate)))
        u_prime.append({**u_entry, "data": u_raw[:n] - moving_mean(u_raw[:n], window)})
        w_prime.append({**w_entry, "data": w_raw[:n] - moving_mean(w_raw[:n], window)})
    artifacts: list[PlotArtifact] = []
    if flags["plot_quadrant_scatter"]:
        figure = plot_quadrant_scatter(u_prime, w_prime, f"{config.site.upper()}: quadrant scatter")
        artifacts.append(artifact_from_figure(config, "quadrant_scatter", figure, flags["save_figures"]))
    if flags["plot_quadrant_joint_pdf"]:
        figure = plot_quadrant_joint_pdf(u_prime, w_prime, f"{config.site.upper()}: quadrant joint PDF")
        artifacts.append(artifact_from_figure(config, "quadrant_joint_pdf", figure, flags["save_figures"]))
    return artifacts


def main() -> None:
    args = driver_parser(__doc__ or "Quadrant analysis").parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
