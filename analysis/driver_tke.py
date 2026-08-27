"""TKE and TKE-transport driver."""

from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from pathlib import Path

from analysis.driver_common import align_height_series, driver_parser, load_data, load_driver_config, plot_height_series, resolve_flags, variance_height_series
from analysis.models import PlotArtifact
from analysis.tke import compute_tke_transport_flux, turbulent_kinetic_energy
from tools.common import get_variable_time_axis
from tools.series import collect_height_series

PLOT_TKE = False
PLOT_TKE_TRANSPORT = False
PLOT_FULL_TKE_BUDGET = False
SAVE_FIGURES = False


def run(config_path: str | Path | None = None, flag_overrides: dict[str, bool] | None = None) -> list[PlotArtifact]:
    flags = resolve_flags(
        {
            "plot_tke": PLOT_TKE,
            "plot_tke_transport": PLOT_TKE_TRANSPORT,
            "plot_full_tke_budget": PLOT_FULL_TKE_BUDGET,
            "save_figures": SAVE_FIGURES,
        }, flag_overrides,
    )
    if flags["plot_full_tke_budget"]:
        raise NotImplementedError(
            "Unsupported TKE flag: plot_full_tke_budget. The full TKE budget is a documented placeholder."
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
    return artifacts


def main() -> None:
    args = driver_parser(__doc__ or "TKE analysis").parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
