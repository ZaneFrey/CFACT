"""Radiation time-series driver."""

from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from pathlib import Path

from analysis.driver_common import (
    driver_parser,
    load_data,
    load_driver_config,
    mean_height_series,
    plot_height_series,
    resolve_flags,
)
from analysis.models import PlotArtifact

PLOT_RADIATION = False
SAVE_FIGURES = True


def run(config_path: str | Path | None = None, flag_overrides: dict[str, bool] | None = None) -> list[PlotArtifact]:
    flags = resolve_flags(
        {
            "plot_radiation": PLOT_RADIATION,
            "save_figures": SAVE_FIGURES,
        },
        flag_overrides,
    )
    if not flags["plot_radiation"]:
        return []
    config = load_driver_config(config_path)
    data = load_data(config, ["Rsw_in", "Rsw_out", "Rlw_in", "Rlw_out"], cadence="5min")
    artifacts: list[PlotArtifact] = []
    for prefix, name, title in (
        ("Rsw_in", "shortwave_incoming", "Incoming shortwave radiation"),
        ("Rsw_out", "shortwave_outgoing", "Outgoing shortwave radiation"),
        ("Rlw_in", "longwave_incoming", "Incoming longwave radiation"),
        ("Rlw_out", "longwave_outgoing", "Outgoing longwave radiation"),
    ):
        time_axis, series = mean_height_series(data, config, prefix)
        artifacts.append(
            plot_height_series(
                config,
                name,
                f"{config.site.upper()}: {title}",
                r"Radiation [W m$^{-2}$]",
                time_axis,
                series,
                save_figures=flags["save_figures"],
            )
        )
    return artifacts


def main() -> None:
    args = driver_parser(__doc__ or "Radiation analysis").parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
