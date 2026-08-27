"""Meteorological time-series driver."""

from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from pathlib import Path
from typing import Any

from analysis.driver_common import (
    align_height_series,
    driver_parser,
    load_data,
    load_driver_config,
    mean_height_series,
    plot_height_series,
    resolve_flags,
    wind_direction,
)
from analysis.models import PlotArtifact

PLOT_U = False
PLOT_V = False
PLOT_W = False
PLOT_WIND_SPEED = True
PLOT_WIND_DIRECTION = True
PLOT_SONIC_TEMPERATURE = True
PLOT_AMBIENT_TEMPERATURE = False
PLOT_RELATIVE_HUMIDITY = False
SAVE_FIGURES = True


def run(config_path: str | Path | None = None, flag_overrides: dict[str, bool] | None = None) -> list[PlotArtifact]:
    defaults = {
        "plot_u": PLOT_U,
        "plot_v": PLOT_V,
        "plot_w": PLOT_W,
        "plot_wind_speed": PLOT_WIND_SPEED,
        "plot_wind_direction": PLOT_WIND_DIRECTION,
        "plot_sonic_temperature": PLOT_SONIC_TEMPERATURE,
        "plot_ambient_temperature": PLOT_AMBIENT_TEMPERATURE,
        "plot_relative_humidity": PLOT_RELATIVE_HUMIDITY,
        "save_figures": SAVE_FIGURES,
    }
    flags = resolve_flags(defaults, flag_overrides)
    config = load_driver_config(config_path)
    enabled = [name for name, value in flags.items() if name.startswith("plot_") and value]
    if not enabled:
        return []
    prefixes: list[str] = []
    for flag, needed in {
        "plot_u": ["u"],
        "plot_v": ["v"],
        "plot_w": ["w"],
        "plot_wind_speed": ["spd"],
        "plot_wind_direction": ["u", "v"],
        "plot_sonic_temperature": ["tc"],
        "plot_ambient_temperature": ["T"],
        "plot_relative_humidity": ["RH"],
    }.items():
        if flags[flag]:
            prefixes.extend(needed)
    data = load_data(config, list(dict.fromkeys(prefixes))) if prefixes else None
    artifacts: list[PlotArtifact] = []
    save = flags["save_figures"]
    specifications = [
        ("plot_u", "u", "u", "Mean u", r"u [m s$^{-1}$]", None),
        ("plot_v", "v", "v", "Mean v", r"v [m s$^{-1}$]", None),
        ("plot_w", "w", "w", "Mean w", r"w [m s$^{-1}$]", None),
        ("plot_wind_speed", "spd", "wind_speed", "Wind speed", r"Speed [m s$^{-1}$]", None),
        ("plot_sonic_temperature", "tc", "sonic_temperature", "Sonic temperature", r"Temperature [$^\circ$C]", None),
        ("plot_ambient_temperature", "T", "ambient_temperature", "Ambient temperature", r"Temperature [$^\circ$C]", None),
        ("plot_relative_humidity", "RH", "relative_humidity", "Relative humidity", "RH [%]", (0.0, 110.0)),
    ]
    for flag, prefix, name, title, ylabel, limits in specifications:
        if not flags[flag]:
            continue
        assert data is not None
        time_axis, series = mean_height_series(data, config, prefix)
        artifacts.append(
            plot_height_series(
                config, name, f"{config.site.upper()}: {title}", ylabel, time_axis, series,
                y_limits=limits, save_figures=save,
            )
        )
    if flags["plot_wind_direction"]:
        assert data is not None
        time_axis, u_series = mean_height_series(data, config, "u")
        _, v_series = mean_height_series(data, config, "v")
        u_series, v_series = align_height_series(u_series, v_series)
        direction_series: list[dict[str, Any]] = []
        for u_entry, v_entry in zip(u_series, v_series):
            direction_series.append({**u_entry, "data": wind_direction(u_entry["data"], v_entry["data"])})
        artifacts.append(
            plot_height_series(
                config,
                "wind_direction",
                f"{config.site.upper()}: Wind direction",
                "Direction [degrees from north]",
                time_axis,
                direction_series,
                y_limits=(0.0, 360.0),
                save_figures=save,
            )
        )
    return artifacts


def main() -> None:
    args = driver_parser(__doc__ or "Meteorological analysis").parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
