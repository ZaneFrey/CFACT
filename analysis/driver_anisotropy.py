"""Turbulence-anisotropy driver."""

from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from pathlib import Path

import numpy as np

from analysis.anisotropy import barycentric_coordinates
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
from analysis.spectra import compute_1d_spectrum
from tools.common import get_variable_time_axis
from tools.plotting import plot_energy_spectra, plot_lumley_trajectory, plot_lumley_triangle
from tools.series import collect_height_series

PLOT_X_B = True
PLOT_Y_B = True
PLOT_ANISOTROPY_SPECTRA = False
PLOT_LUMLEY_TRIANGLE = False
PLOT_TRAJECTORIES = False
PLOT_TRIANGLE_ANIMATION = False
SAVE_FIGURES = True
TRAJECTORY_HEIGHT_METERS = 2.0


def run(config_path: str | Path | None = None, flag_overrides: dict[str, bool] | None = None) -> list[PlotArtifact]:
    flags = resolve_flags(
        {
            "plot_x_b": PLOT_X_B,
            "plot_y_b": PLOT_Y_B,
            "plot_anisotropy_spectra": PLOT_ANISOTROPY_SPECTRA,
            "plot_lumley_triangle": PLOT_LUMLEY_TRIANGLE,
            "plot_trajectories": PLOT_TRAJECTORIES,
            "plot_triangle_animation": PLOT_TRIANGLE_ANIMATION,
            "save_figures": SAVE_FIGURES,
        },
        flag_overrides,
    )
    if flags["plot_triangle_animation"]:
        raise NotImplementedError(
            "Triangle animation is a documented placeholder. Enable x_B/y_B, spectra, Lumley triangle, or trajectories instead."
        )
    if not any(value for name, value in flags.items() if name.startswith("plot_")):
        return []
    config = load_driver_config(config_path)
    data = load_data(config, ["u", "v", "w"])
    _, time_local, _, _ = get_variable_time_axis(data)
    u_series = collect_height_series(data, config.site, "u")
    v_series = collect_height_series(data, config.site, "v")
    w_series = collect_height_series(data, config.site, "w")
    u_series, v_series, w_series = align_height_series(u_series, v_series, w_series)
    x_series = []
    y_series = []
    bary_time = None
    for u_entry, v_entry, w_entry in zip(u_series, v_series, w_series):
        bary_time, x_b, y_b = barycentric_coordinates(
            u_entry["data"], u_entry["meta"],
            v_entry["data"], v_entry["meta"],
            w_entry["data"], w_entry["meta"],
            time_local, config.averaging_period_seconds, config.centered_gliding,
        )
        x_series.append({**u_entry, "data": x_b, "time": bary_time, "varName": "x_b"})
        y_series.append({**u_entry, "data": y_b, "time": bary_time, "varName": "y_b"})
    if bary_time is None:
        raise ValueError("No shared u/v/w height series were found.")
    artifacts: list[PlotArtifact] = []
    if flags["plot_x_b"]:
        artifacts.append(
            plot_height_series(
                config, "anisotropy_x_b", f"{config.site.upper()}: Anisotropy $x_B$", "$x_B$",
                bary_time, x_series, y_limits=(0.0, 1.0), save_figures=flags["save_figures"],
            )
        )
    if flags["plot_y_b"]:
        artifacts.append(
            plot_height_series(
                config, "anisotropy_y_b", f"{config.site.upper()}: Anisotropy $y_B$", "$y_B$",
                bary_time, y_series, y_limits=(0.0, float(np.sqrt(3) / 2)), save_figures=flags["save_figures"],
            )
        )
    if flags["plot_anisotropy_spectra"]:
        spectra_stats = []
        for name, label, source in (("x_b", "$x_B$", x_series), ("y_b", "$y_B$", y_series)):
            output = []
            for entry in source:
                frequency, spectrum = compute_1d_spectrum(entry["data"], None, bary_time, apply_log_binning=True)
                output.append({**entry, "frequencyHz": frequency, "data": spectrum})
            spectra_stats.append({"varName": name, "displayName": label, "series": output})
        figure = plot_energy_spectra(spectra_stats, f"{config.site.upper()}: anisotropy spectra")
        artifacts.append(artifact_from_figure(config, "anisotropy_spectra", figure, flags["save_figures"]))
    if flags["plot_lumley_triangle"]:
        figure = plot_lumley_triangle(x_series, y_series, f"{config.site.upper()}: Lumley triangle")
        artifacts.append(artifact_from_figure(config, "lumley_triangle", figure, flags["save_figures"]))
    if flags["plot_trajectories"]:
        index = int(np.argmin([abs(float(entry["heightValue"]) - TRAJECTORY_HEIGHT_METERS) for entry in x_series]))
        figure = plot_lumley_trajectory(
            [x_series[index]], [y_series[index]], bary_time,
            f"{config.site.upper()}: anisotropy trajectory at {x_series[index]['heightValue']:g} m",
        )
        artifacts.append(artifact_from_figure(config, "anisotropy_trajectory", figure, flags["save_figures"]))
    return artifacts


def main() -> None:
    args = driver_parser(__doc__ or "Anisotropy analysis").parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
