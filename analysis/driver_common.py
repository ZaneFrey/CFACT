"""Shared orchestration used by independently runnable analysis drivers."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.config import AnalysisConfig, load_config
from analysis.models import PlotArtifact
from analysis.statistics import compute_window_stat
from tools.common import build_height_colormap, format_height_label, get_variable_time_axis
from tools.figures import save_figure
from tools.netcdf import read_fluxes
from tools.selection import select_files_by_local_timerange
from tools.series import collect_height_series
from tools.style import apply_style


def resolve_flags(defaults: Mapping[str, bool], overrides: Mapping[str, bool] | None) -> dict[str, bool]:
    flags = dict(defaults)
    if overrides:
        unknown = set(overrides) - set(flags)
        if unknown:
            raise ValueError(f"Unknown flag override(s): {', '.join(sorted(unknown))}")
        for name, value in overrides.items():
            if not isinstance(value, (bool, np.bool_)):
                raise TypeError(f"Flag {name!r} must be boolean.")
            flags[name] = bool(value)
    return flags


def selected_files(config: AnalysisConfig, cadence: str) -> list[str]:
    selected, _ = select_files_by_local_timerange(
        config.data_dir, config.start_time, config.end_time, config.timezone
    )
    token = "_hr_" if cadence == "20hz" else "_5min_"
    files = [name for name in selected if token in Path(name).name.lower()]
    if not files:
        raise FileNotFoundError(f"No {cadence} NetCDF files overlap the configured interval.")
    return files


def load_data(config: AnalysisConfig, prefixes: list[str], cadence: str = "20hz") -> dict[str, Any]:
    return read_fluxes(
        selected_files(config, cadence),
        site_codes=[config.site],
        var_prefixes=prefixes,
        local_timezone=config.timezone,
        start_time_local=config.start_time,
        end_time_local=config.end_time,
    )


def mean_height_series(data: dict[str, Any], config: AnalysisConfig, prefix: str) -> tuple[pd.DatetimeIndex, list[dict[str, Any]]]:
    _, time_local, _, _ = get_variable_time_axis(data)
    output: list[dict[str, Any]] = []
    output_time = pd.DatetimeIndex([])
    for entry in collect_height_series(data, config.site, prefix):
        time_stat, values = compute_window_stat(
            entry["data"],
            entry["meta"],
            time_local,
            config.averaging_period_seconds,
            config.centered_gliding,
            "mean",
        )
        output.append({**entry, "data": values, "time": time_stat})
        output_time = time_stat
    return output_time, output


def variance_height_series(data: dict[str, Any], config: AnalysisConfig, prefix: str) -> tuple[pd.DatetimeIndex, list[dict[str, Any]]]:
    _, time_local, _, _ = get_variable_time_axis(data)
    output: list[dict[str, Any]] = []
    output_time = pd.DatetimeIndex([])
    for entry in collect_height_series(data, config.site, prefix):
        time_stat, values = compute_window_stat(
            entry["data"],
            entry["meta"],
            time_local,
            config.averaging_period_seconds,
            config.centered_gliding,
            "var",
        )
        output.append({**entry, "data": values, "time": time_stat})
        output_time = time_stat
    return output_time, output


def align_height_series(*groups: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    if not groups:
        return []
    common = {entry["heightTag"] for entry in groups[0]}
    for group in groups[1:]:
        common &= {entry["heightTag"] for entry in group}
    order = [entry["heightTag"] for entry in groups[0] if entry["heightTag"] in common]
    return [[{entry["heightTag"]: entry for entry in group}[tag] for tag in order] for group in groups]


def wind_direction(u: Any, v: Any) -> np.ndarray:
    """Meteorological direction from which the mean wind is blowing."""

    u_values, v_values = np.broadcast_arrays(np.asarray(u, dtype=float), np.asarray(v, dtype=float))
    return np.mod(np.degrees(np.arctan2(-u_values, -v_values)), 360.0)


def plot_height_series(
    config: AnalysisConfig,
    name: str,
    title: str,
    ylabel: str,
    time_axis: Any,
    series: list[dict[str, Any]],
    *,
    y_limits: tuple[float, float] | None = None,
    save_figures: bool,
) -> PlotArtifact:
    if not series:
        raise ValueError(f"No height series were available for {name}.")
    apply_style(config.figure)
    ordered = sorted(series, key=lambda entry: float(entry["heightValue"]))
    colors = build_height_colormap(len(ordered), config.figure.height_colormap)
    fig, ax = plt.subplots(figsize=(config.figure.width, config.figure.panel_height + 1.1))
    finite_series = 0
    for color, entry in zip(colors, ordered):
        values = np.asarray(entry["data"], dtype=float).reshape(-1)
        times = pd.DatetimeIndex(entry.get("time", time_axis))
        if values.size != times.size:
            raise ValueError(f"Time/data length mismatch for {entry['heightTag']} in {name}.")
        finite = np.isfinite(values)
        if np.any(finite):
            finite_series += 1
            ax.plot(times[finite], values[finite], color=color, label=format_height_label(entry["heightTag"]))
    if finite_series == 0:
        plt.close(fig)
        raise ValueError(f"No finite plotted data were produced for {name}.")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(f"Local time ({config.timezone})")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=config.start_time.tzinfo))
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, title="Height")
    fig.tight_layout()
    return save_figure(
        name,
        fig,
        config.output_dir,
        config.figure.output_format,
        config.figure.dpi,
        config.figure.overwrite,
        save_figures,
    )


def artifact_from_figure(
    config: AnalysisConfig, name: str, figure, save_figures: bool
) -> PlotArtifact:
    return save_figure(
        name,
        figure,
        config.output_dir,
        config.figure.output_format,
        config.figure.dpi,
        config.figure.overwrite,
        save_figures,
    )


def driver_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, default=None, help="Path to an analysis YAML configuration.")
    return parser


def load_driver_config(config_path: str | Path | None) -> AnalysisConfig:
    return load_config(config_path)
