"""Integral-timescale-analysis driver."""

from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from pathlib import Path

from analysis.driver_common import (
    align_height_series,
    driver_parser,
    load_data,
    load_driver_config,
    plot_height_series,
    resolve_flags,
)
from analysis.models import PlotArtifact
from analysis.statistics import compute_integral_timescale, compute_window_stat
from tools.common import get_variable_time_axis
from tools.series import collect_height_series

PLOT_INTEGRAL_TIMESCALE = True
SAVE_FIGURES = True

INTEGRAL_TIMESCALE_PAIR = ("u", "u")
MAX_LAG_SECONDS = 900


def run(config_path: str | Path | None = None, flag_overrides: dict[str, bool] | None = None) -> list[PlotArtifact]:
    flags = resolve_flags(
        {
            "plot_integral_timescale": PLOT_INTEGRAL_TIMESCALE,
            "save_figures": SAVE_FIGURES,
        },
        flag_overrides,
    )
    if not flags["plot_integral_timescale"]:
        return []
    config = load_driver_config(config_path)
    prefix_x, prefix_y = INTEGRAL_TIMESCALE_PAIR
    data = load_data(
        config,
        list(dict.fromkeys(INTEGRAL_TIMESCALE_PAIR)),
        time_padding_seconds=MAX_LAG_SECONDS / 2.0,
    )
    _, time_local, _, _ = get_variable_time_axis(data)
    x_series = collect_height_series(data, config.site, prefix_x)
    y_series = collect_height_series(data, config.site, prefix_y)
    x_series, y_series = align_height_series(x_series, y_series)
    output = []
    output_time = None
    for x_entry, y_entry in zip(x_series, y_series):
        raw_time, raw_values = compute_integral_timescale(
            x_entry["data"],
            x_entry["meta"],
            y_entry["data"],
            y_entry["meta"],
            time_local,
            MAX_LAG_SECONDS,
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
    return [
        plot_height_series(
            config,
            "integral_timescale",
            f"{config.site.upper()}: integral timescale",
            "Integral timescale [s]",
            output_time,
            output,
            save_figures=flags["save_figures"],
        )
    ]


def main() -> None:
    args = driver_parser(__doc__ or "Integral timescale analysis").parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
