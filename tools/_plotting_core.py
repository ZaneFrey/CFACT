from __future__ import annotations

import math
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tools.common import build_height_colormap, collect_master_height_tags, format_height_label, height_tag_to_value


def _align_series_by_tag(series_a: list[dict[str, Any]], series_b: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tags_a = [item["heightTag"] for item in series_a]
    tags_b = [item["heightTag"] for item in series_b]
    common = [tag for tag in tags_a if tag in tags_b]
    if not common:
        raise ValueError("No shared height tags were found between the two series groups.")
    out_a = [series_a[tags_a.index(tag)] for tag in common]
    out_b = [series_b[tags_b.index(tag)] for tag in common]
    return out_a, out_b


def _lookup_height_color(height_tag: str, master_tags: list[str], cmap: np.ndarray) -> np.ndarray:
    try:
        idx = master_tags.index(height_tag)
    except ValueError:
        idx = 0
    return cmap[idx]


def _variable_title(name: str) -> str:
    lookup = {"u": "u", "v": "v", "w": "w", "t": "Temperature"}
    return lookup.get(str(name).lower(), str(name))


def _variable_units(name: str) -> str:
    lookup = {"u": "[m s$^{-1}$]", "v": "[m s$^{-1}$]", "w": "[m s$^{-1}$]", "t": "[$^\\circ$C]"}
    return lookup.get(str(name).lower(), "")


def _format_variable_label(name: str, is_variance: bool) -> str:
    base = {
        "u": "u",
        "v": "v",
        "w": "w",
        "t": "T",
        "theta": r"$\theta$",
        "theta_v": r"$\theta_v$",
        "q": "q",
        "tau_int": r"$T_i$",
        "rh": "RH",
        "xB": r"$x_B$",
        "yB": r"$y_B$",
    }.get(str(name), str(name))
    if not is_variance:
        return base
    return rf"$\sigma^2_{{{base}}}$"


def _valid_mask(x: Any, y: Any) -> np.ndarray:
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    valid = np.isfinite(y_arr)
    if np.issubdtype(np.asarray(x).dtype, np.number):
        valid &= np.isfinite(np.asarray(x, dtype=float).reshape(-1))
    return valid


def plot_stacked_height_stats(stats: list[dict[str, Any]], figTitle: str | None = None, isVariance: bool = False):
    if not stats:
        raise ValueError("The stats input cannot be empty.")
    master_tags = collect_master_height_tags(stats)
    cmap = build_height_colormap(max(len(master_tags), 1))
    fig, axes = plt.subplots(len(stats), 1, figsize=(12.4, 2.4 * len(stats) + 0.6), squeeze=False)
    axes_flat = axes[:, 0]
    if figTitle:
        fig.suptitle(figTitle, fontweight="bold")
    legend_ax = None
    legend_tags: list[str] = []
    for ax, stat in zip(axes_flat, stats):
        ax.grid(True)
        ax.set_title(stat.get("displayName", stat.get("varName", "")))
        for series in stat.get("series", []):
            x = np.asarray(series.get("time", stat.get("time", np.arange(np.asarray(series["data"]).size))))
            y = np.asarray(series["data"], dtype=float).reshape(-1)
            if x.size != y.size:
                continue
            valid = _valid_mask(x, y)
            if not np.any(valid):
                continue
            tag = str(series.get("heightTag", "single"))
            ax.plot(x[valid], y[valid], lw=1.2, color=_lookup_height_color(tag, master_tags, cmap))
            if legend_ax is None:
                legend_ax = ax
            if tag not in legend_tags:
                legend_tags.append(tag)
        if "yLimits" in stat and stat["yLimits"] is not None and len(stat["yLimits"]) == 2:
            ax.set_ylim(stat["yLimits"])
        ax.set_ylabel(_format_variable_label(stat.get("varName", stat.get("displayName", "")), isVariance))
    axes_flat[-1].set_xlabel("Local Time" if hasattr(stats[0].get("time", []), "tz") else "Sample")
    if legend_ax is not None and legend_tags:
        handles = [
            legend_ax.plot([], [], lw=1.5, color=_lookup_height_color(tag, master_tags, cmap), label=format_height_label(tag))[0]
            for tag in legend_tags
        ]
        legend_ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, title="Height")
    fig.tight_layout()
    return fig


def _add_minus_five_thirds(ax, frequency_hz: np.ndarray, spectral_density: np.ndarray) -> None:
    valid = np.isfinite(frequency_hz) & np.isfinite(spectral_density) & (frequency_hz > 0) & (spectral_density > 0)
    if np.count_nonzero(valid) < 2:
        return
    f = frequency_hz[valid]
    s = spectral_density[valid]
    log_f = np.log10(f)
    log_s = np.log10(s)
    f1 = 10 ** (np.min(log_f) + 0.55 * (np.max(log_f) - np.min(log_f)))
    f2 = 10 ** (np.min(log_f) + 0.9 * (np.max(log_f) - np.min(log_f)))
    anchor = 10 ** (np.min(log_s) + 0.55 * (np.max(log_s) - np.min(log_s)))
    s_ref = anchor * (np.array([f1, f2]) / f1) ** (-5.0 / 3.0)
    ax.loglog([f1, f2], s_ref, "k--", lw=1.0)


def plot_energy_spectra(stats: list[dict[str, Any]], figTitle: str | None = None):
    if not stats:
        raise ValueError("The stats input cannot be empty.")
    master_tags = collect_master_height_tags(stats)
    cmap = build_height_colormap(len(master_tags))
    fig, axes = plt.subplots(1, len(stats), figsize=(4.0 * len(stats), 4.8), squeeze=False)
    axes_flat = axes[0]
    if figTitle:
        fig.suptitle(figTitle, fontweight="bold")
    for ax, stat in zip(axes_flat, stats):
        panel_f = []
        panel_s = []
        ax.grid(True)
        for series in stat.get("series", []):
            f = np.asarray(series.get("frequencyHz", stat.get("frequencyHz", [])), dtype=float).reshape(-1)
            s = np.asarray(series.get("data", []), dtype=float).reshape(-1)
            valid = np.isfinite(f) & np.isfinite(s) & (f > 0) & (s > 0)
            if not np.any(valid):
                continue
            tag = str(series.get("heightTag", "single"))
            ax.loglog(f[valid], s[valid], lw=1.4, color=_lookup_height_color(tag, master_tags, cmap))
            panel_f.append(f[valid])
            panel_s.append(s[valid])
        if panel_f:
            _add_minus_five_thirds(ax, np.concatenate(panel_f), np.concatenate(panel_s))
        else:
            ax.text(0.5, 0.5, "No data found", transform=ax.transAxes, ha="center", va="center", color="0.4")
        ax.set_title(_variable_title(stat.get("displayName", stat.get("varName", ""))))
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Spectral Energy Density")
        ax.set_xscale("log")
        ax.set_yscale("log")
    handles = [axes_flat[0].plot([], [], lw=1.5, color=_lookup_height_color(tag, master_tags, cmap), label=format_height_label(tag))[0] for tag in master_tags]
    axes_flat[0].legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, title="Height")
    fig.tight_layout()
    return fig


def plot_cospectra(stats: list[dict[str, Any]], figTitle: str | None = None):
    if not stats:
        raise ValueError("The stats input cannot be empty.")
    master_tags = collect_master_height_tags(stats)
    cmap = build_height_colormap(len(master_tags))
    fig, axes = plt.subplots(2, len(stats), figsize=(4.0 * len(stats), 5.6), squeeze=False)
    if figTitle:
        fig.suptitle(figTitle, fontweight="bold")
    for col, stat in enumerate(stats):
        ax_norm = axes[0, col]
        ax_signed = axes[1, col]
        ax_norm.grid(True)
        ax_signed.grid(True)
        any_norm = False
        any_signed = False
        signed_values = []
        freq_values = []
        for series in stat.get("series", []):
            f = np.asarray(series.get("frequencyHz", stat.get("frequencyHz", [])), dtype=float).reshape(-1)
            c = np.asarray(series.get("data", []), dtype=float).reshape(-1)
            tag = str(series.get("heightTag", "single"))
            color = _lookup_height_color(tag, master_tags, cmap)
            signed = f * c
            valid_signed = np.isfinite(f) & np.isfinite(signed) & (f > 0)
            if np.any(valid_signed):
                ax_signed.semilogx(f[valid_signed], signed[valid_signed], lw=1.4, color=color)
                any_signed = True
                signed_values.append(signed[valid_signed])
                freq_values.append(f[valid_signed])
            normalized = np.asarray(series.get("normalizedData", []), dtype=float).reshape(-1)
            valid_norm = np.isfinite(f) & np.isfinite(normalized) & (f > 0) & (normalized > 0)
            if np.any(valid_norm):
                ax_norm.loglog(f[valid_norm], normalized[valid_norm], lw=1.4, color=color)
                any_norm = True
        if not any_norm:
            ax_norm.text(0.5, 0.5, "No positive normalized data to plot", transform=ax_norm.transAxes, ha="center", va="center", color="0.4")
        if not any_signed:
            ax_signed.text(0.5, 0.5, "No finite data to plot", transform=ax_signed.transAxes, ha="center", va="center", color="0.4")
        ax_signed.axhline(0.0, color="k", lw=0.8)
        ax_norm.set_title(stat.get("displayName", stat.get("varName", "")))
        ax_norm.set_ylabel("Normalized\n|f Co(f)| / |x'y'|")
        ax_signed.set_ylabel("f Co(f)")
        ax_signed.set_xlabel("Frequency [Hz]")
        if freq_values:
            f_cat = np.concatenate(freq_values)
            ax_norm.set_xlim(np.nanmin(f_cat), np.nanmax(f_cat))
            ax_signed.set_xlim(np.nanmin(f_cat), np.nanmax(f_cat))
        if signed_values:
            s_cat = np.concatenate(signed_values)
            lim = max(1.0, float(np.nanmax(np.abs(s_cat))))
            ax_signed.set_ylim(-1.05 * lim, 1.05 * lim)
    handles = [axes[0, 0].plot([], [], lw=1.5, color=_lookup_height_color(tag, master_tags, cmap), label=format_height_label(tag))[0] for tag in master_tags]
    axes[0, 0].legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, title="Height")
    fig.tight_layout()
    return fig


def plot_autocorrelation(stats: list[dict[str, Any]], figTitle: str | None = None):
    if not stats:
        raise ValueError("The stats input cannot be empty.")
    master_tags = collect_master_height_tags(stats)
    cmap = build_height_colormap(len(master_tags))
    fig, axes = plt.subplots(1, len(stats), figsize=(4.0 * len(stats), 2.8), squeeze=False)
    if figTitle:
        fig.suptitle(figTitle, fontweight="bold")
    for ax, stat in zip(axes[0], stats):
        lag_seconds = np.asarray(stat.get("lagSeconds", []), dtype=float).reshape(-1)
        lag_plot = lag_seconds / 60.0 if lag_seconds.size and lag_seconds[-1] >= 120 else lag_seconds
        ax.grid(True)
        for series in stat.get("series", []):
            y = np.asarray(series.get("data", []), dtype=float).reshape(-1)
            valid = np.isfinite(lag_plot) & np.isfinite(y)
            if not np.any(valid):
                continue
            tag = str(series.get("heightTag", "single"))
            ax.plot(lag_plot[valid], y[valid], lw=1.4, color=_lookup_height_color(tag, master_tags, cmap))
        ax.axhline(0.0, color="0.4", lw=0.8, ls=":")
        ax.set_ylim(-0.2, 1.0)
        ax.set_title(_variable_title(stat.get("displayName", stat.get("varName", ""))))
        ax.set_xlabel("Lag [min]" if lag_seconds.size and lag_seconds[-1] >= 120 else "Lag [s]")
        ax.set_ylabel("R_AA(L)")
    handles = [axes[0, 0].plot([], [], lw=1.5, color=_lookup_height_color(tag, master_tags, cmap), label=format_height_label(tag))[0] for tag in master_tags]
    axes[0, 0].legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, title="Height")
    fig.tight_layout()
    return fig


def _build_histogram_edges(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.array([-0.5, 0.5])
    if values.size == 1 or np.nanmax(values) == np.nanmin(values):
        delta = max(abs(values[0]) * 0.05, 0.5)
        return np.array([values[0] - delta, values[0] + delta])
    edges = np.histogram_bin_edges(values, bins="fd")
    if edges.size < 2:
        return np.linspace(np.nanmin(values), np.nanmax(values), 31)
    return edges


def _compute_moments(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return float("nan"), float("nan")
    prime = values - np.nanmean(values)
    m2 = np.nanmean(prime**2)
    if not np.isfinite(m2) or m2 <= 0:
        return float("nan"), float("nan")
    return float(np.nanmean(prime**3) / (m2 ** 1.5)), float(np.nanmean(prime**4) / (m2**2))


def plot_height_histograms(stats: list[dict[str, Any]], figTitle: str | None = None):
    if not stats:
        raise ValueError("The stats input cannot be empty.")
    master_tags = collect_master_height_tags(stats)
    cmap = build_height_colormap(len(master_tags))
    fig, axes = plt.subplots(2, len(stats), figsize=(4.0 * len(stats), 5.6), squeeze=False)
    if figTitle:
        fig.suptitle(figTitle, fontweight="bold")
    for col, stat in enumerate(stats):
        ax_pdf = axes[0, col]
        ax_cdf = axes[1, col]
        all_samples = []
        samples_per_series: list[np.ndarray] = []
        for series in stat.get("series", []):
            sample = np.asarray(series.get("data", []), dtype=float).reshape(-1)
            sample = sample[np.isfinite(sample)]
            samples_per_series.append(sample)
            if sample.size:
                all_samples.append(sample)
        pooled = np.concatenate(all_samples) if all_samples else np.asarray([])
        edges = _build_histogram_edges(pooled)
        for series, sample in zip(stat.get("series", []), samples_per_series):
            tag = str(series.get("heightTag", "single"))
            color = _lookup_height_color(tag, master_tags, cmap)
            if sample.size:
                ax_pdf.hist(sample, bins=edges, density=True, histtype="step", lw=1.5, color=color)
                x_cdf = np.sort(sample)
                y_cdf = np.arange(1, x_cdf.size + 1, dtype=float) / x_cdf.size
                ax_cdf.plot(x_cdf, y_cdf, lw=1.5, color=color)
            skewness, kurtosis = _compute_moments(sample)
            ax_pdf.text(
                0.98,
                0.98 - 0.075 * list(stat.get("series", [])).index(series),
                f"{format_height_label(tag)}: S = {skewness:.3f}, K = {kurtosis:.3f}",
                transform=ax_pdf.transAxes,
                ha="right",
                va="top",
                fontsize=8.5,
                family="Consolas",
                color=color,
            )
        ax_pdf.set_title(_variable_title(stat.get("displayName", stat.get("varName", ""))))
        ax_pdf.set_xlabel(_variable_units(stat.get("displayName", stat.get("varName", ""))))
        ax_pdf.set_ylabel("PDF")
        ax_cdf.set_xlabel(_variable_units(stat.get("displayName", stat.get("varName", ""))))
        ax_cdf.set_ylabel("CDF")
        ax_cdf.set_ylim(0, 1)
        ax_pdf.grid(True)
        ax_cdf.grid(True)
    handles = [axes[0, 0].plot([], [], lw=1.5, color=_lookup_height_color(tag, master_tags, cmap), label=format_height_label(tag))[0] for tag in master_tags]
    axes[0, 0].legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, title="Height")
    fig.tight_layout()
    return fig


def plot_spectral_panel(spectraStats: list[dict[str, Any]], cospectraStats: list[dict[str, Any]], figTitle: str | None = None):
    if len(spectraStats) != 4 or len(cospectraStats) != 4:
        raise ValueError("The spectral panel requires 4 spectra panels and 4 co-spectra panels.")
    master_tags = collect_master_height_tags(list(spectraStats) + list(cospectraStats))
    cmap = build_height_colormap(len(master_tags))
    fig, axes = plt.subplots(3, 4, figsize=(12.4, 8.4), squeeze=False)
    if figTitle:
        fig.suptitle(figTitle, fontweight="bold")
    for col in range(4):
        top = axes[0, col]
        mid = axes[1, col]
        bot = axes[2, col]
        top.grid(True)
        mid.grid(True)
        bot.grid(True)
        stat = spectraStats[col]
        plot_energy_spectra([stat], None).axes[0].remove()
        panel_f = []
        panel_s = []
        for series in stat.get("series", []):
            f = np.asarray(series.get("frequencyHz", stat.get("frequencyHz", [])), dtype=float).reshape(-1)
            s = np.asarray(series.get("data", []), dtype=float).reshape(-1)
            valid = np.isfinite(f) & np.isfinite(s) & (f > 0) & (s > 0)
            if not np.any(valid):
                continue
            color = _lookup_height_color(str(series.get("heightTag", "single")), master_tags, cmap)
            top.loglog(f[valid], s[valid], lw=1.4, color=color)
            panel_f.append(f[valid])
            panel_s.append(s[valid])
        if panel_f:
            _add_minus_five_thirds(top, np.concatenate(panel_f), np.concatenate(panel_s))
        top.set_title(_variable_title(stat.get("displayName", stat.get("varName", ""))))
        top.set_ylabel("Spectral Energy Density" if col == 0 else "")
        cstat = cospectraStats[col]
        for series in cstat.get("series", []):
            f = np.asarray(series.get("frequencyHz", cstat.get("frequencyHz", [])), dtype=float).reshape(-1)
            normalized = np.asarray(series.get("normalizedData", []), dtype=float).reshape(-1)
            signed = f * np.asarray(series.get("data", []), dtype=float).reshape(-1)
            color = _lookup_height_color(str(series.get("heightTag", "single")), master_tags, cmap)
            valid_n = np.isfinite(f) & np.isfinite(normalized) & (f > 0) & (normalized > 0)
            valid_s = np.isfinite(f) & np.isfinite(signed) & (f > 0)
            if np.any(valid_n):
                mid.loglog(f[valid_n], normalized[valid_n], lw=1.4, color=color)
            if np.any(valid_s):
                bot.semilogx(f[valid_s], signed[valid_s], lw=1.4, color=color)
        mid.set_title(cstat.get("displayName", cstat.get("varName", "")))
        mid.set_ylabel("Normalized\n|f Co(f)| / |x'y'|" if col == 0 else "")
        bot.axhline(0.0, color="k", lw=0.8)
        bot.set_ylabel("f Co(f)" if col == 0 else "")
        bot.set_xlabel("Frequency [Hz]")
    handles = [axes[0, 0].plot([], [], lw=1.5, color=_lookup_height_color(tag, master_tags, cmap), label=format_height_label(tag))[0] for tag in master_tags]
    axes[0, 0].legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, title="Height")
    fig.tight_layout()
    return fig


def plot_wavelet_scalogram(stats: dict[str, Any], figTitle: str | None = None):
    series = list(stats.get("series", []))
    if not series:
        raise ValueError("The stats input must contain at least one height series.")
    fig, axes = plt.subplots(len(series), 1, figsize=(12.4, max(2.8, 2.15 * len(series))), squeeze=False)
    if figTitle:
        fig.suptitle(figTitle, fontweight="bold")
    magnitudes = [np.asarray(entry.get("scalogramMagnitude", []), dtype=float) for entry in series]
    finite_magnitudes = [magnitude[np.isfinite(magnitude)] for magnitude in magnitudes]
    finite_magnitudes = [magnitude for magnitude in finite_magnitudes if magnitude.size]
    if not finite_magnitudes:
        raise ValueError("Wavelet scalogram magnitudes contain no finite values.")
    magnitude_min = min(float(np.min(magnitude)) for magnitude in finite_magnitudes)
    magnitude_max = max(float(np.max(magnitude)) for magnitude in finite_magnitudes)
    if magnitude_min == magnitude_max:
        magnitude_max = float(np.nextafter(magnitude_max, np.inf))
    for ax, entry in zip(axes[:, 0], series):
        frequency_hz = np.asarray(entry.get("frequencyHz", []), dtype=float).reshape(-1)
        magnitude = np.asarray(entry.get("scalogramMagnitude", []), dtype=float)
        time_local = pd.DatetimeIndex(entry.get("scalogramTime", []))
        valid = np.isfinite(frequency_hz) & (frequency_hz > 0)
        frequency_hz = frequency_hz[valid]
        magnitude = magnitude[valid, :]
        time_num = mdates.date2num(time_local.to_pydatetime())
        mesh = ax.pcolormesh(
            time_num,
            frequency_hz,
            magnitude,
            shading="auto",
            cmap="turbo",
            vmin=magnitude_min,
            vmax=magnitude_max,
        )
        ax.set_yscale("log")
        ax.set_ylabel("f [Hz]")
        ax.set_title(format_height_label(str(entry.get("heightTag", "single"))), fontweight="normal")
        fig.colorbar(mesh, ax=ax, label="|WT|")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=time_local.tz))
    axes[-1, 0].set_xlabel("Local Time")
    fig.tight_layout()
    return fig


def plot_lumley_triangle(xBSeries: list[dict[str, Any]], yBSeries: list[dict[str, Any]], figTitle: str | None = None):
    if len(xBSeries) != len(yBSeries):
        raise ValueError("xBSeries and yBSeries must contain the same number of heights.")
    xBSeries, yBSeries = _align_series_by_tag(list(xBSeries), list(yBSeries))
    master_tags = [entry["heightTag"] for entry in xBSeries]
    cmap = build_height_colormap(max(len(master_tags), 1))
    fig, ax = plt.subplots(figsize=(9.0, 7.6))
    if figTitle:
        ax.set_title(figTitle)
    triangle_x = [0, 1, 0.5, 0]
    triangle_y = [0, 0, math.sqrt(3) / 2, 0]
    ax.plot(triangle_x, triangle_y, "k-", lw=1.4)
    for idx, (x_entry, y_entry) in enumerate(zip(xBSeries, yBSeries)):
        x = np.asarray(x_entry.get("data", []), dtype=float).reshape(-1)
        y = np.asarray(y_entry.get("data", []), dtype=float).reshape(-1)
        valid = np.isfinite(x) & np.isfinite(y)
        ax.scatter(
            x[valid],
            y[valid],
            s=28,
            c=[cmap[idx]],
            edgecolors="none",
            alpha=0.65,
            label=format_height_label(master_tags[idx]),
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, math.sqrt(3) / 2)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xlabel("$x_B$")
    ax.set_ylabel("$y_B$")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, title="Height")
    fig.tight_layout()
    return fig


def plot_lumley_trajectory(xBSeries: list[dict[str, Any]], yBSeries: list[dict[str, Any]], timeAxis: Any, figTitle: str | None = None):
    if len(xBSeries) != 1 or len(yBSeries) != 1:
        raise ValueError("The Lumley trajectory plot requires exactly one height in xBSeries and yBSeries.")
    x_entry = list(xBSeries)[0]
    y_entry = list(yBSeries)[0]
    x = np.asarray(x_entry.get("data", []), dtype=float).reshape(-1)
    y = np.asarray(y_entry.get("data", []), dtype=float).reshape(-1)
    time_local = np.asarray(timeAxis)
    if not (x.size == y.size == time_local.size):
        raise ValueError("The Lumley trajectory plot requires x_B, y_B, and time to have the same length.")
    valid = np.isfinite(x) & np.isfinite(y)
    if not np.any(valid):
        raise ValueError("The selected Lumley trajectory contains no finite barycentric coordinates.")
    x = x[valid]
    y = y[valid]
    time_local = time_local[valid]
    color_progress = np.linspace(0.0, 1.0, x.size) if x.size > 1 else np.asarray([0.0])
    fig, ax = plt.subplots(figsize=(9.0, 7.6))
    if figTitle:
        ax.set_title(figTitle)
    triangle_x = [0, 1, 0.5, 0]
    triangle_y = [0, 0, math.sqrt(3) / 2, 0]
    ax.plot(triangle_x, triangle_y, "k-", lw=1.4)
    scatter = ax.scatter(x, y, c=color_progress, cmap=plt.get_cmap("berlin"), s=28, edgecolors="none", alpha=0.85)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, math.sqrt(3) / 2)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xlabel("$x_B$")
    ax.set_ylabel("$y_B$")
    colorbar = fig.colorbar(scatter, ax=ax, label="Time")
    if time_local.size == 1:
        colorbar.set_ticks([0.0])
        colorbar.set_ticklabels([mdates.num2date(mdates.date2num(time_local[0])).strftime("%H:%M")])
    else:
        tick_positions = np.linspace(0.0, 1.0, min(3, time_local.size))
        tick_indices = np.linspace(0, time_local.size - 1, tick_positions.size, dtype=int)
        tick_labels = [mdates.num2date(mdates.date2num(time_local[idx])).strftime("%H:%M") for idx in tick_indices]
        colorbar.set_ticks(tick_positions)
        colorbar.set_ticklabels(tick_labels)
    fig.tight_layout()
    return fig


def _determine_symmetric_axis_limit(values: Any) -> float:
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 1.0
    return max(1.0, 1.05 * float(np.nanmax(np.abs(arr))))


def _annotate_quadrants(ax, axis_limit: float) -> None:
    label_offset = 0.83 * axis_limit
    ax.text(label_offset, label_offset, "QI: Outward", ha="right", va="top", color="0.25", fontsize=10)
    ax.text(-label_offset, label_offset, "QII: Ejection", ha="left", va="top", color="0.25", fontsize=10)
    ax.text(-label_offset, -label_offset, "QIII: Inward", ha="left", va="bottom", color="0.25", fontsize=10)
    ax.text(label_offset, -label_offset, "QIV: Sweep", ha="right", va="bottom", color="0.25", fontsize=10)


def plot_quadrant_scatter(uPrimeSeries: list[dict[str, Any]], wPrimeSeries: list[dict[str, Any]], figTitle: str | None = None):
    if len(uPrimeSeries) != len(wPrimeSeries):
        raise ValueError("uPrimeSeries and wPrimeSeries must contain the same number of heights.")
    uPrimeSeries, wPrimeSeries = _align_series_by_tag(list(uPrimeSeries), list(wPrimeSeries))
    master_tags = [entry["heightTag"] for entry in uPrimeSeries]
    cmap = build_height_colormap(max(len(master_tags), 1))
    fig, ax = plt.subplots(figsize=(6.8, 6.8))
    axis_data = []
    for idx, (u_entry, w_entry) in enumerate(zip(uPrimeSeries, wPrimeSeries)):
        u_prime = np.asarray(u_entry.get("data", []), dtype=float).reshape(-1)
        w_prime = np.asarray(w_entry.get("data", []), dtype=float).reshape(-1)
        valid = np.isfinite(u_prime) & np.isfinite(w_prime)
        if np.any(valid):
            axis_data.extend([*u_prime[valid], *w_prime[valid]])
        ax.scatter(
            u_prime[valid],
            w_prime[valid],
            s=18,
            c=[cmap[idx]],
            edgecolors="none",
            alpha=0.32,
            label=format_height_label(master_tags[idx]),
        )
    axis_limit = _determine_symmetric_axis_limit(axis_data)
    ax.set_xlim(-axis_limit, axis_limit)
    ax.set_ylim(-axis_limit, axis_limit)
    ax.set_aspect("equal")
    ax.grid(True)
    ax.axhline(0.0, color="k", lw=0.9)
    ax.axvline(0.0, color="k", lw=0.9)
    _annotate_quadrants(ax, axis_limit)
    ax.set_xlabel("u'/σ_u")
    ax.set_ylabel("w'/σ_w")
    ax.set_title(figTitle or "Quadrant analysis")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, title="Height")
    fig.tight_layout()
    return fig


def plot_quadrant_joint_pdf(uPrimeSeries: list[dict[str, Any]], wPrimeSeries: list[dict[str, Any]], figTitle: str | None = None):
    if len(uPrimeSeries) != len(wPrimeSeries):
        raise ValueError("uPrimeSeries and wPrimeSeries must contain the same number of heights.")
    uPrimeSeries, wPrimeSeries = _align_series_by_tag(list(uPrimeSeries), list(wPrimeSeries))
    master_tags = [entry["heightTag"] for entry in uPrimeSeries]
    cmap = build_height_colormap(max(len(master_tags), 1))
    fig, ax = plt.subplots(figsize=(6.8, 6.8))
    axis_data = []
    for u_entry, w_entry in zip(uPrimeSeries, wPrimeSeries):
        u_prime = np.asarray(u_entry.get("data", []), dtype=float).reshape(-1)
        w_prime = np.asarray(w_entry.get("data", []), dtype=float).reshape(-1)
        valid = np.isfinite(u_prime) & np.isfinite(w_prime)
        if np.any(valid):
            axis_data.extend([*u_prime[valid], *w_prime[valid]])
    axis_limit = _determine_symmetric_axis_limit(axis_data)
    probability_levels = [0.95, 0.90, 0.75, 0.50]
    line_widths = [1.2, 1.6, 2.1, 2.7]
    for idx, (u_entry, w_entry) in enumerate(zip(uPrimeSeries, wPrimeSeries)):
        u_prime = np.asarray(u_entry.get("data", []), dtype=float).reshape(-1)
        w_prime = np.asarray(w_entry.get("data", []), dtype=float).reshape(-1)
        valid = np.isfinite(u_prime) & np.isfinite(w_prime)
        if not np.any(valid):
            ax.plot([], [], color=cmap[idx], lw=line_widths[-1], label=format_height_label(master_tags[idx]))
            continue
        n_samples = int(np.count_nonzero(valid))
        n_bins = max(40, min(90, round(math.sqrt(max(n_samples, 1)) / 4)))
        edges = np.linspace(-axis_limit, axis_limit, n_bins + 1)
        hist, xedges, yedges = np.histogram2d(u_prime[valid], w_prime[valid], bins=[edges, edges], density=False)
        prob = hist / np.sum(hist) if np.sum(hist) > 0 else hist
        mass = prob[prob > 0]
        thresholds = []
        if mass.size:
            mass_sorted = np.sort(mass)[::-1]
            cumulative = np.cumsum(mass_sorted)
            for level in probability_levels:
                position = np.searchsorted(cumulative, level, side="left")
                position = min(position, mass_sorted.size - 1)
                thresholds.append(mass_sorted[position])
        else:
            thresholds = [np.nan] * len(probability_levels)
        xcenters = 0.5 * (xedges[:-1] + xedges[1:])
        ycenters = 0.5 * (yedges[:-1] + yedges[1:])
        representative = None
        for level, width, threshold in zip(probability_levels, line_widths, thresholds):
            if not np.isfinite(threshold) or threshold <= 0:
                continue
            contour = ax.contour(xcenters, ycenters, prob.T, levels=[threshold], colors=[cmap[idx]], linewidths=width)
            if representative is None and contour.collections:
                representative = contour.collections[0]
        if representative is not None:
            representative.set_label(format_height_label(master_tags[idx]))
        else:
            ax.plot([], [], color=cmap[idx], lw=line_widths[-1], label=format_height_label(master_tags[idx]))
    ax.set_xlim(-axis_limit, axis_limit)
    ax.set_ylim(-axis_limit, axis_limit)
    ax.set_aspect("equal")
    ax.grid(True)
    ax.axhline(0.0, color="k", lw=0.9)
    ax.axvline(0.0, color="k", lw=0.9)
    _annotate_quadrants(ax, axis_limit)
    ax.set_xlabel("u'/σ_u")
    ax.set_ylabel("w'/σ_w")
    ax.set_title(figTitle or "Quadrant joint PDF")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, title="Height")
    fig.tight_layout()
    return fig
