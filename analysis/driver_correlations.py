"""Correlation, integral-timescale, and quadrant-analysis driver."""

from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from pathlib import Path

import numpy as np

from analysis._math import expand_raw_series, moving_mean
from analysis.driver_common import (
    align_height_series,
    artifact_from_figure,
    driver_parser,
    load_data,
    load_driver_config,
    plot_height_series,
    resolve_flags,
)
from analysis.models import PlotArtifact
from analysis.statistics import compute_autocorrelation, compute_integral_timescale, compute_window_stat
from tools.common import get_variable_time_axis
from tools.plotting import plot_autocorrelation, plot_quadrant_joint_pdf, plot_quadrant_scatter
from tools.series import collect_height_series
from tools.style import apply_style

PLOT_AUTOCORRELATION = False
PLOT_INTEGRAL_TIMESCALE = True
PLOT_QUADRANT_SCATTER = False
PLOT_QUADRANT_JOINT_PDF = False
SAVE_FIGURES = True

AUTOCORRELATION_COMPONENTS = ("u", "v", "w", "tc")
INTEGRAL_TIMESCALE_PAIR = ("u", "u")
MAX_LAG_SECONDS = 900


def run(config_path: str | Path | None = None, flag_overrides: dict[str, bool] | None = None) -> list[PlotArtifact]:
    flags = resolve_flags(
        {
            "plot_autocorrelation": PLOT_AUTOCORRELATION,
            "plot_integral_timescale": PLOT_INTEGRAL_TIMESCALE,
            "plot_quadrant_scatter": PLOT_QUADRANT_SCATTER,
            "plot_quadrant_joint_pdf": PLOT_QUADRANT_JOINT_PDF,
            "save_figures": SAVE_FIGURES,
        }, flag_overrides,
    )
    if not any(value for name, value in flags.items() if name.startswith("plot_")):
        return []
    config = load_driver_config(config_path)
    prefixes = list(AUTOCORRELATION_COMPONENTS if flags["plot_autocorrelation"] else ())
    if flags["plot_quadrant_scatter"] or flags["plot_quadrant_joint_pdf"]:
        prefixes.extend(["u", "w"])
    data = load_data(config, list(dict.fromkeys(prefixes))) if prefixes else None
    time_local = None
    if data is not None:
        _, time_local, _, _ = get_variable_time_axis(data)
    apply_style(config.figure)
    artifacts: list[PlotArtifact] = []
    if flags["plot_autocorrelation"]:
        assert data is not None and time_local is not None
        stats = []
        for prefix in AUTOCORRELATION_COMPONENTS:
            output = []
            lags = np.array([])
            for entry in collect_height_series(data, config.site, prefix):
                lags, values = compute_autocorrelation(entry["data"], entry["meta"], time_local, MAX_LAG_SECONDS)
                output.append({**entry, "data": values})
            stats.append({"varName": prefix, "displayName": prefix, "lagSeconds": lags, "series": output})
        figure = plot_autocorrelation(stats, f"{config.site.upper()}: autocorrelation")
        artifacts.append(artifact_from_figure(config, "autocorrelation", figure, flags["save_figures"]))
    if flags["plot_integral_timescale"]:
        prefix_x, prefix_y = INTEGRAL_TIMESCALE_PAIR
        integral_data = load_data(
            config,
            list(dict.fromkeys(INTEGRAL_TIMESCALE_PAIR)),
            time_padding_seconds=MAX_LAG_SECONDS / 2.0,
        )
        _, integral_time_local, _, _ = get_variable_time_axis(integral_data)
        x_series = collect_height_series(integral_data, config.site, prefix_x)
        y_series = collect_height_series(integral_data, config.site, prefix_y)
        x_series, y_series = align_height_series(x_series, y_series)
        output = []
        output_time = None
        for x_entry, y_entry in zip(x_series, y_series):
            raw_time, raw_values = compute_integral_timescale(
                x_entry["data"], x_entry["meta"], y_entry["data"], y_entry["meta"],
                integral_time_local, MAX_LAG_SECONDS,
            )
            requested = (raw_time >= config.start_time) & (raw_time <= config.end_time)
            output_time, values = compute_window_stat(
                raw_values[requested],
                None,
                raw_time[requested],
                config.averaging_period_seconds,
                config.centered_gliding,
                "mean",
            )
            output.append({**x_entry, "data": values, "time": output_time})
        artifacts.append(
            plot_height_series(
                config, "integral_timescale", f"{config.site.upper()}: integral timescale",
                "Integral timescale [s]", output_time, output, save_figures=flags["save_figures"],
            )
        )
    if flags["plot_quadrant_scatter"] or flags["plot_quadrant_joint_pdf"]:
        assert data is not None and time_local is not None
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
            up = u_raw[:n] - moving_mean(u_raw[:n], window)
            wp = w_raw[:n] - moving_mean(w_raw[:n], window)
            u_prime.append({**u_entry, "data": up})
            w_prime.append({**w_entry, "data": wp})
        if flags["plot_quadrant_scatter"]:
            figure = plot_quadrant_scatter(u_prime, w_prime, f"{config.site.upper()}: quadrant scatter")
            artifacts.append(artifact_from_figure(config, "quadrant_scatter", figure, flags["save_figures"]))
        if flags["plot_quadrant_joint_pdf"]:
            figure = plot_quadrant_joint_pdf(u_prime, w_prime, f"{config.site.upper()}: quadrant joint PDF")
            artifacts.append(artifact_from_figure(config, "quadrant_joint_pdf", figure, flags["save_figures"]))
    return artifacts


def main() -> None:
    args = driver_parser(__doc__ or "Correlation analysis").parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
