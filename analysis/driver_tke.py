"""TKE, friction-velocity, turbulent-flux, and transport driver."""

from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from pathlib import Path

from analysis.driver_common import align_height_series, driver_parser, load_data, load_driver_config, plot_height_series, resolve_flags, variance_height_series
from analysis.models import PlotArtifact
from analysis.statistics import compute_window_covariance
from analysis.thermodynamics import specific_humidity_flux
from analysis.tke import compute_tke_transport_flux, friction_velocity, turbulent_kinetic_energy
from tools.common import get_variable_time_axis
from tools.series import collect_height_series

PLOT_TKE = False
PLOT_FRICTION_VELOCITY = False
PLOT_REYNOLDS_FLUXES = False
PLOT_MOISTURE_FLUXES = False
PLOT_TKE_TRANSPORT = False
PLOT_FULL_TKE_BUDGET = False
PLOT_Z_OVER_L = False
SAVE_FIGURES = False


def _covariance_series(data, config, prefix_x, prefix_y):
    _, time_local, _, _ = get_variable_time_axis(data)
    x_series = collect_height_series(data, config.site, prefix_x)
    y_series = collect_height_series(data, config.site, prefix_y)
    x_series, y_series = align_height_series(x_series, y_series)
    output = []
    output_time = None
    for x_entry, y_entry in zip(x_series, y_series):
        output_time, values = compute_window_covariance(
            x_entry["data"], x_entry["meta"], y_entry["data"], y_entry["meta"], time_local,
            config.averaging_period_seconds, config.centered_gliding,
        )
        output.append({**x_entry, "data": values, "time": output_time})
    return output_time, output


def run(config_path: str | Path | None = None, flag_overrides: dict[str, bool] | None = None) -> list[PlotArtifact]:
    flags = resolve_flags(
        {
            "plot_tke": PLOT_TKE,
            "plot_friction_velocity": PLOT_FRICTION_VELOCITY,
            "plot_reynolds_fluxes": PLOT_REYNOLDS_FLUXES,
            "plot_moisture_fluxes": PLOT_MOISTURE_FLUXES,
            "plot_tke_transport": PLOT_TKE_TRANSPORT,
            "plot_full_tke_budget": PLOT_FULL_TKE_BUDGET,
            "plot_z_over_l": PLOT_Z_OVER_L,
            "save_figures": SAVE_FIGURES,
        }, flag_overrides,
    )
    unsupported = [name for name in ("plot_full_tke_budget", "plot_z_over_l") if flags[name]]
    if unsupported:
        raise NotImplementedError(
            f"Unsupported TKE flag(s): {', '.join(unsupported)}. Full TKE budget and z/L are documented placeholders."
        )
    if not any(value for name, value in flags.items() if name.startswith("plot_")):
        return []
    config = load_driver_config(config_path)
    data = load_data(config, ["u", "v", "w"])
    artifacts: list[PlotArtifact] = []
    save = flags["save_figures"]
    if flags["plot_tke"]:
        time_axis, var_u = variance_height_series(data, config, "u")
        _, var_v = variance_height_series(data, config, "v")
        _, var_w = variance_height_series(data, config, "w")
        var_u, var_v, var_w = align_height_series(var_u, var_v, var_w)
        series = [{**u, "data": turbulent_kinetic_energy(u["data"], v["data"], w["data"])} for u, v, w in zip(var_u, var_v, var_w)]
        artifacts.append(plot_height_series(config, "tke", f"{config.site.upper()}: TKE", r"TKE [m$^2$ s$^{-2}$]", time_axis, series, save_figures=save))
    if flags["plot_friction_velocity"]:
        time_axis, uw = _covariance_series(data, config, "u", "w")
        _, vw = _covariance_series(data, config, "v", "w")
        uw, vw = align_height_series(uw, vw)
        series = [{**u, "data": friction_velocity(u["data"], v["data"])} for u, v in zip(uw, vw)]
        artifacts.append(plot_height_series(config, "friction_velocity", f"{config.site.upper()}: friction velocity", r"$u_*$ [m s$^{-1}$]", time_axis, series, save_figures=save))
    if flags["plot_reynolds_fluxes"]:
        for px, py in (("u", "w"), ("v", "w"), ("w", "w")):
            time_axis, series = _covariance_series(data, config, px, py)
            artifacts.append(plot_height_series(config, f"reynolds_flux_{px}_{py}", f"{config.site.upper()}: {px}{py} flux", rf"$\overline{{{px}'{py}'}}$", time_axis, series, save_figures=save))
    if flags["plot_tke_transport"]:
        _, time_local, _, _ = get_variable_time_axis(data)
        u = collect_height_series(data, config.site, "u")
        v = collect_height_series(data, config.site, "v")
        w = collect_height_series(data, config.site, "w")
        u, v, w = align_height_series(u, v, w)
        series = []
        output_time = None
        for eu, ev, ew in zip(u, v, w):
            output_time, values = compute_tke_transport_flux(
                eu["data"], eu["meta"], ev["data"], ev["meta"], ew["data"], ew["meta"], time_local,
                config.averaging_period_seconds, config.centered_gliding,
            )
            series.append({**eu, "data": values, "time": output_time})
        artifacts.append(plot_height_series(config, "tke_transport", f"{config.site.upper()}: vertical TKE transport", r"$\overline{w'e}$", output_time, series, save_figures=save))
    if flags["plot_moisture_fluxes"]:
        moisture = load_data(config, ["w_h2o_", "h2o", "T", "P"], cadence="5min")
        _, moisture_time, _, _ = get_variable_time_axis(moisture)
        flux = collect_height_series(moisture, config.site, "w_h2o_")
        vapor = collect_height_series(moisture, config.site, "h2o")
        temperature = collect_height_series(moisture, config.site, "T")
        pressure = collect_height_series(moisture, config.site, "P")
        flux, vapor, temperature, pressure = align_height_series(flux, vapor, temperature, pressure)
        series = []
        for flux_entry, vapor_entry, temperature_entry, pressure_entry in zip(flux, vapor, temperature, pressure):
            _, w_q = specific_humidity_flux(
                vapor_entry["data"], temperature_entry["data"], pressure_entry["data"], flux_entry["data"]
            )
            series.append({**flux_entry, "data": w_q, "time": moisture_time})
        artifacts.append(
            plot_height_series(
                config, "moisture_flux", f"{config.site.upper()}: moisture flux", r"$\overline{w'q'}$ [kg kg$^{-1}$ m s$^{-1}$]",
                moisture_time, series, save_figures=save,
            )
        )
    return artifacts


def main() -> None:
    args = driver_parser(__doc__ or "TKE analysis").parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
