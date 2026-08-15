"""Windowed statistics, correlations, and turbulence diagnostics."""

from analysis._math import (
    compute_autocorrelation,
    compute_integral_timescale,
    compute_window_covariance,
    compute_window_stat,
)

__all__ = [
    "compute_autocorrelation",
    "compute_integral_timescale",
    "compute_window_covariance",
    "compute_window_stat",
]
