"""Snake-case plotting interfaces shared by analysis drivers."""

from __future__ import annotations

from typing import Any

from tools import _plotting_core


def plot_stacked_height_stats(stats: list[dict[str, Any]], fig_title: str | None = None, is_variance: bool = False):
    return _plotting_core.plot_stacked_height_stats(stats, fig_title, is_variance)


def plot_energy_spectra(stats: list[dict[str, Any]], fig_title: str | None = None):
    return _plotting_core.plot_energy_spectra(stats, fig_title)


def plot_cospectra(stats: list[dict[str, Any]], fig_title: str | None = None):
    return _plotting_core.plot_cospectra(stats, fig_title)


def plot_autocorrelation(stats: list[dict[str, Any]], fig_title: str | None = None):
    return _plotting_core.plot_autocorrelation(stats, fig_title)


def plot_height_histograms(stats: list[dict[str, Any]], fig_title: str | None = None):
    return _plotting_core.plot_height_histograms(stats, fig_title)


def plot_spectral_panel(spectra_stats: list[dict[str, Any]], cospectra_stats: list[dict[str, Any]], fig_title: str | None = None):
    return _plotting_core.plot_spectral_panel(spectra_stats, cospectra_stats, fig_title)


def plot_wavelet_scalogram(stats: dict[str, Any], fig_title: str | None = None):
    return _plotting_core.plot_wavelet_scalogram(stats, fig_title)


def plot_lumley_triangle(x_b_series: list[dict[str, Any]], y_b_series: list[dict[str, Any]], fig_title: str | None = None):
    return _plotting_core.plot_lumley_triangle(x_b_series, y_b_series, fig_title)


def plot_lumley_trajectory(x_b_series: list[dict[str, Any]], y_b_series: list[dict[str, Any]], time_axis: Any, fig_title: str | None = None):
    return _plotting_core.plot_lumley_trajectory(x_b_series, y_b_series, time_axis, fig_title)


def plot_quadrant_scatter(u_prime_series: list[dict[str, Any]], w_prime_series: list[dict[str, Any]], fig_title: str | None = None):
    return _plotting_core.plot_quadrant_scatter(u_prime_series, w_prime_series, fig_title)


def plot_quadrant_joint_pdf(u_prime_series: list[dict[str, Any]], w_prime_series: list[dict[str, Any]], fig_title: str | None = None):
    return _plotting_core.plot_quadrant_joint_pdf(u_prime_series, w_prime_series, fig_title)


__all__ = [name for name in globals() if name.startswith("plot_")]
