"""Simplified turbulent-kinetic-energy budget driver."""

from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
from analysis.statistics import compute_window_covariance, compute_window_stat
from analysis.tke import (
    compute_dissipation_rate,
    compute_tke_transport_flux,
    simplified_tke_budget,
    turbulent_kinetic_energy,
)
from tools.common import get_variable_time_axis
from tools.series import collect_height_series
from tools.style import apply_style

PLOT_TOTAL_TKE = True
PLOT_HEIGHT = 3.0
PLOT_STORAGE = True
PLOT_TKE_ADVEC = True
PLOT_BUOY_PROD = True
PLOT_SHEAR_PROD = True
PLOT_TKE_TRANSPORT = True
PLOT_DISSIPATION = True
PLOT_RESIDUAL = True
PLOT_BUDGET = True
SAVE_FIGURES = True

_TERM_DETAILS = {
    "storage": ("tke_storage", "TKE storage", r"$\partial\bar{e}/\partial t$"),
    "tke_advection": ("tke_advection", "TKE advection", r"$-\bar{w}\,\partial\bar{e}/\partial z$"),
    "buoyancy_production": (
        "buoyancy_production",
        "Buoyancy production",
        r"$(g/\bar{\theta}_v)\overline{w'\theta_v'}$",
    ),
    "shear_production": (
        "shear_production",
        "Shear production",
        r"$-\overline{u_s'w'}\,\partial\bar{u}_s/\partial z$",
    ),
    "tke_transport": ("tke_transport", "Turbulent TKE transport", r"$-\partial\overline{w'e}/\partial z$"),
    "dissipation": ("tke_dissipation", "TKE dissipation", r"$-\epsilon$"),
    "residual": ("tke_residual", "TKE budget residual", "Residual (including pressure transport)"),
}


def _as_height_time(rows: list[np.ndarray]) -> np.ndarray:
    return np.vstack([np.asarray(row, dtype=float).reshape(-1) for row in rows])


def _compute_budget_inputs(
    data: dict[str, Any], config
) -> tuple[pd.DatetimeIndex, list[dict[str, Any]], np.ndarray, dict[str, np.ndarray]]:
    _, time_local, _, _ = get_variable_time_axis(data)
    u = collect_height_series(data, config.site, "u")
    v = collect_height_series(data, config.site, "v")
    w = collect_height_series(data, config.site, "w")
    virtual_temperature = collect_height_series(data, config.site, "tc")
    u, v, w, virtual_temperature = align_height_series(u, v, w, virtual_temperature)
    if len(u) < 3:
        raise ValueError(
            f"The TKE budget requires at least 3 common u/v/w/tc heights; found {len(u)} at site {config.site!r}."
        )

    time_output: pd.DatetimeIndex | None = None
    tke_rows: list[np.ndarray] = []
    mean_u_rows: list[np.ndarray] = []
    mean_w_rows: list[np.ndarray] = []
    covariance_uw_rows: list[np.ndarray] = []
    mean_virtual_temperature_rows: list[np.ndarray] = []
    virtual_temperature_flux_rows: list[np.ndarray] = []
    tke_flux_rows: list[np.ndarray] = []
    dissipation_rows: list[np.ndarray] = []

    for u_entry, v_entry, w_entry, temperature_entry in zip(u, v, w, virtual_temperature):
        time_stat, variance_u = compute_window_stat(
            u_entry["data"], u_entry["meta"], time_local,
            config.averaging_period_seconds, config.centered_gliding, "var",
        )
        _, variance_v = compute_window_stat(
            v_entry["data"], v_entry["meta"], time_local,
            config.averaging_period_seconds, config.centered_gliding, "var",
        )
        _, variance_w = compute_window_stat(
            w_entry["data"], w_entry["meta"], time_local,
            config.averaging_period_seconds, config.centered_gliding, "var",
        )
        _, mean_u = compute_window_stat(
            u_entry["data"], u_entry["meta"], time_local,
            config.averaging_period_seconds, config.centered_gliding, "mean",
        )
        _, mean_w = compute_window_stat(
            w_entry["data"], w_entry["meta"], time_local,
            config.averaging_period_seconds, config.centered_gliding, "mean",
        )
        _, covariance_uw = compute_window_covariance(
            u_entry["data"], u_entry["meta"], w_entry["data"], w_entry["meta"], time_local,
            config.averaging_period_seconds, config.centered_gliding,
        )
        # tc is virtual air temperature from the sonic. Adding 273.15 changes
        # its mean to kelvin while leaving its covariance with w unchanged.
        _, mean_virtual_temperature = compute_window_stat(
            np.asarray(temperature_entry["data"], dtype=float) + 273.15,
            temperature_entry["meta"], time_local,
            config.averaging_period_seconds, config.centered_gliding, "mean",
        )
        _, virtual_temperature_flux = compute_window_covariance(
            w_entry["data"], w_entry["meta"],
            temperature_entry["data"], temperature_entry["meta"], time_local,
            config.averaging_period_seconds, config.centered_gliding,
        )
        _, tke_flux = compute_tke_transport_flux(
            u_entry["data"], u_entry["meta"],
            v_entry["data"], v_entry["meta"],
            w_entry["data"], w_entry["meta"], time_local,
            config.averaging_period_seconds, config.centered_gliding,
        )
        _, epsilon = compute_dissipation_rate(
            u_entry["data"], u_entry["meta"], time_local,
            config.averaging_period_seconds, config.centered_gliding,
        )
        if time_output is None:
            time_output = time_stat
        elif not time_output.equals(time_stat):
            raise ValueError("TKE inputs produced inconsistent output time axes.")
        tke_rows.append(turbulent_kinetic_energy(variance_u, variance_v, variance_w))
        mean_u_rows.append(mean_u)
        mean_w_rows.append(mean_w)
        covariance_uw_rows.append(covariance_uw)
        mean_virtual_temperature_rows.append(mean_virtual_temperature)
        virtual_temperature_flux_rows.append(virtual_temperature_flux)
        tke_flux_rows.append(tke_flux)
        dissipation_rows.append(epsilon)

    assert time_output is not None
    heights = np.asarray([entry["heightValue"] for entry in u], dtype=float)
    tke_values = _as_height_time(tke_rows)
    terms = simplified_tke_budget(
        time_output,
        heights,
        tke_values,
        _as_height_time(mean_u_rows),
        _as_height_time(mean_w_rows),
        _as_height_time(covariance_uw_rows),
        _as_height_time(mean_virtual_temperature_rows),
        _as_height_time(virtual_temperature_flux_rows),
        _as_height_time(tke_flux_rows),
        _as_height_time(dissipation_rows),
    )
    return time_output, u, tke_values, terms


def _selected_height_index(heights: np.ndarray) -> int:
    requested = float(PLOT_HEIGHT)
    matches = np.flatnonzero(np.isclose(heights, requested, rtol=0.0, atol=1e-9))
    available = ", ".join(f"{height:g}" for height in heights)
    if not matches.size:
        raise ValueError(f"PLOT_HEIGHT={requested:g} m is unavailable. Common heights are: {available} m.")
    index = int(matches[0])
    if index == 0 or index == heights.size - 1:
        raise ValueError(
            f"PLOT_HEIGHT must be an intermediate measurement height; {requested:g} m is a boundary height."
        )
    return index


def _plot_term(
    config,
    time: pd.DatetimeIndex,
    height_m: float,
    term_name: str,
    values: np.ndarray,
    save_figures: bool,
) -> PlotArtifact:
    artifact_name, title, label = _TERM_DETAILS[term_name]
    apply_style(config.figure)
    fig, ax = plt.subplots(figsize=(config.figure.width, config.figure.panel_height + 1.1))
    ax.plot(time, values, color=plt.get_cmap("turbo")(0.55), label=label)
    ax.set_title(f"{config.site.upper()}: {title} at {height_m:g} m")
    ax.set_ylabel(r"$\partial\bar{e}/\partial t$ [m$^2$ s$^{-3}$]")
    ax.set_xlabel(f"Local time ({config.timezone})")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=config.start_time.tzinfo))
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.tight_layout()
    return artifact_from_figure(config, artifact_name, fig, save_figures)


def _plot_budget(
    config,
    time: pd.DatetimeIndex,
    height_m: float,
    terms: dict[str, np.ndarray],
    height_index: int,
    save_figures: bool,
) -> PlotArtifact:
    apply_style(config.figure)
    fig, ax = plt.subplots(figsize=(config.figure.width, config.figure.panel_height + 1.1))
    names = list(_TERM_DETAILS)
    colors = plt.get_cmap("turbo")(np.linspace(0.04, 0.96, len(names)))
    for color, name in zip(colors, names):
        ax.plot(time, terms[name][height_index], color=color, label=_TERM_DETAILS[name][2])
    ax.set_title(f"{config.site.upper()}: simplified TKE budget at {height_m:g} m")
    ax.set_ylabel(r"$\partial\bar{e}/\partial t$ [m$^2$ s$^{-3}$]")
    ax.set_xlabel(f"Local time ({config.timezone})")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=config.start_time.tzinfo))
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.tight_layout()
    return artifact_from_figure(config, "tke_budget", fig, save_figures)


def run(
    config_path: str | Path | None = None,
    flag_overrides: dict[str, bool] | None = None,
) -> list[PlotArtifact]:
    flags = resolve_flags(
        {
            "plot_total_tke": PLOT_TOTAL_TKE,
            "plot_storage": PLOT_STORAGE,
            "plot_tke_advec": PLOT_TKE_ADVEC,
            "plot_buoy_prod": PLOT_BUOY_PROD,
            "plot_shear_prod": PLOT_SHEAR_PROD,
            "plot_tke_transport": PLOT_TKE_TRANSPORT,
            "plot_dissipation": PLOT_DISSIPATION,
            "plot_residual": PLOT_RESIDUAL,
            "plot_budget": PLOT_BUDGET,
            "save_figures": SAVE_FIGURES,
        },
        flag_overrides,
    )
    plot_requested = any(value for name, value in flags.items() if name.startswith("plot_"))
    if not plot_requested:
        return []

    config = load_driver_config(config_path)
    budget_requested = any(
        flags[name]
        for name in (
            "plot_storage", "plot_tke_advec", "plot_buoy_prod", "plot_shear_prod",
            "plot_tke_transport", "plot_dissipation", "plot_residual", "plot_budget",
        )
    )
    prefixes = ["u", "v", "w", "tc"] if budget_requested else ["u", "v", "w"]
    data = load_data(config, prefixes)
    save = flags["save_figures"]
    artifacts: list[PlotArtifact] = []

    if budget_requested:
        time_output, height_entries, tke_values, terms = _compute_budget_inputs(data, config)
    else:
        _, time_local, _, _ = get_variable_time_axis(data)
        u = collect_height_series(data, config.site, "u")
        v = collect_height_series(data, config.site, "v")
        w = collect_height_series(data, config.site, "w")
        u, v, w = align_height_series(u, v, w)
        height_entries = u
        tke_rows = []
        time_output = pd.DatetimeIndex([])
        for u_entry, v_entry, w_entry in zip(u, v, w):
            time_output, variance_u = compute_window_stat(
                u_entry["data"], u_entry["meta"], time_local,
                config.averaging_period_seconds, config.centered_gliding, "var",
            )
            _, variance_v = compute_window_stat(
                v_entry["data"], v_entry["meta"], time_local,
                config.averaging_period_seconds, config.centered_gliding, "var",
            )
            _, variance_w = compute_window_stat(
                w_entry["data"], w_entry["meta"], time_local,
                config.averaging_period_seconds, config.centered_gliding, "var",
            )
            tke_rows.append(turbulent_kinetic_energy(variance_u, variance_v, variance_w))
        tke_values = _as_height_time(tke_rows)
        terms = {}

    if flags["plot_total_tke"]:
        series = [
            {**entry, "data": tke_values[index], "time": time_output}
            for index, entry in enumerate(height_entries)
        ]
        artifacts.append(
            plot_height_series(
                config,
                "total_tke",
                f"{config.site.upper()}: turbulent kinetic energy",
                r"$\bar{e}$ [m$^2$ s$^{-2}$]",
                time_output,
                series,
                save_figures=save,
            )
        )

    if budget_requested:
        heights = np.asarray([entry["heightValue"] for entry in height_entries], dtype=float)
        height_index = _selected_height_index(heights)
        height_m = float(heights[height_index])
        requested_terms = (
            ("plot_storage", "storage"),
            ("plot_tke_advec", "tke_advection"),
            ("plot_buoy_prod", "buoyancy_production"),
            ("plot_shear_prod", "shear_production"),
            ("plot_tke_transport", "tke_transport"),
            ("plot_dissipation", "dissipation"),
            ("plot_residual", "residual"),
        )
        for flag_name, term_name in requested_terms:
            if flags[flag_name]:
                artifacts.append(
                    _plot_term(config, time_output, height_m, term_name, terms[term_name][height_index], save)
                )
        if flags["plot_budget"]:
            artifacts.append(_plot_budget(config, time_output, height_m, terms, height_index, save))
    return artifacts


def main() -> None:
    args = driver_parser(__doc__ or "TKE analysis").parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
