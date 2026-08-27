"""Reynolds and moisture-flux driver."""

from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from pathlib import Path

from analysis.driver_common import (
    align_height_series,
    covariance_height_series,
    driver_parser,
    load_data,
    load_driver_config,
    plot_height_series,
    resolve_flags,
)
from analysis.models import PlotArtifact
from analysis.thermodynamics import specific_humidity_flux
from tools.common import get_variable_time_axis
from tools.series import collect_height_series

PLOT_REYNOLDS_FLUXES = False
PLOT_MOISTURE_FLUXES = False
SAVE_FIGURES = False


def run(config_path: str | Path | None = None, flag_overrides: dict[str, bool] | None = None) -> list[PlotArtifact]:
    flags = resolve_flags(
        {
            "plot_reynolds_fluxes": PLOT_REYNOLDS_FLUXES,
            "plot_moisture_fluxes": PLOT_MOISTURE_FLUXES,
            "save_figures": SAVE_FIGURES,
        },
        flag_overrides,
    )
    if not (flags["plot_reynolds_fluxes"] or flags["plot_moisture_fluxes"]):
        return []
    config = load_driver_config(config_path)
    artifacts: list[PlotArtifact] = []
    if flags["plot_reynolds_fluxes"]:
        data = load_data(config, ["u", "v", "w"])
        for prefix_x, prefix_y in (("u", "w"), ("v", "w"), ("w", "w")):
            time_axis, series = covariance_height_series(data, config, prefix_x, prefix_y)
            artifacts.append(
                plot_height_series(
                    config,
                    f"reynolds_flux_{prefix_x}_{prefix_y}",
                    f"{config.site.upper()}: {prefix_x}{prefix_y} flux",
                    rf"$\overline{{{prefix_x}'{prefix_y}'}}$",
                    time_axis,
                    series,
                    save_figures=flags["save_figures"],
                )
            )
    if flags["plot_moisture_fluxes"]:
        data = load_data(config, ["w_h2o_", "h2o", "T", "P"], cadence="5min")
        _, time_local, _, _ = get_variable_time_axis(data)
        flux = collect_height_series(data, config.site, "w_h2o_")
        vapor = collect_height_series(data, config.site, "h2o")
        temperature = collect_height_series(data, config.site, "T")
        pressure = collect_height_series(data, config.site, "P")
        flux, vapor, temperature, pressure = align_height_series(flux, vapor, temperature, pressure)
        series = []
        for flux_entry, vapor_entry, temperature_entry, pressure_entry in zip(
            flux, vapor, temperature, pressure
        ):
            _, values = specific_humidity_flux(
                vapor_entry["data"],
                temperature_entry["data"],
                pressure_entry["data"],
                flux_entry["data"],
            )
            series.append({**flux_entry, "data": values, "time": time_local})
        artifacts.append(
            plot_height_series(
                config,
                "moisture_flux",
                f"{config.site.upper()}: moisture flux",
                r"$\overline{w'q'}$ [kg kg$^{-1}$ m s$^{-1}$]",
                time_local,
                series,
                save_figures=flags["save_figures"],
            )
        )
    return artifacts


def main() -> None:
    args = driver_parser(__doc__ or "Turbulent flux analysis").parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
