"""Fourier spectra and cospectra."""

from __future__ import annotations

from typing import Any

import numpy as np

from analysis._math import compute_1d_spectrum as _compute_1d_spectrum
from analysis._math import compute_cospectrum as _compute_cospectrum


def compute_1d_spectrum(
    values: Any,
    meta: dict[str, Any] | None = None,
    time_axis: Any = None,
    *,
    apply_log_binning: bool = False,
    log_bins_per_decade: float = 12,
):
    return _compute_1d_spectrum(
        values,
        meta,
        time_axis,
        ApplyLogBinning=apply_log_binning,
        LogBinsPerDecade=log_bins_per_decade,
    )


def compute_cospectrum(
    x: Any,
    meta_x: dict[str, Any] | None,
    y: Any,
    meta_y: dict[str, Any] | None,
    time_axis: Any = None,
    *,
    apply_log_binning: bool = False,
    log_bins_per_decade: float = 12,
):
    return _compute_cospectrum(
        x,
        meta_x,
        y,
        meta_y,
        time_axis,
        ApplyLogBinning=apply_log_binning,
        LogBinsPerDecade=log_bins_per_decade,
    )


def compute_ogive(frequency_hz: Any, cospectral_density: Any) -> np.ndarray:
    """Integrate a cospectrum from the Nyquist frequency toward low frequency.

    The returned values have the same ordering as ``frequency_hz``. At each
    frequency ``f``, the value is the trapezoidal estimate of
    ``integral_f^f_N Co(f') df'``; consequently, the highest-frequency value
    is zero.
    """
    frequency = np.asarray(frequency_hz, dtype=float).reshape(-1)
    cospectrum = np.asarray(cospectral_density, dtype=float).reshape(-1)
    if frequency.size != cospectrum.size:
        raise ValueError("Frequency and cospectral-density arrays must have the same length.")
    if frequency.size == 0:
        return np.asarray([], dtype=float)
    if not np.all(np.isfinite(frequency)) or not np.all(np.isfinite(cospectrum)):
        raise ValueError("Frequency and cospectral-density arrays must contain only finite values.")
    if np.any(frequency <= 0) or np.any(np.diff(frequency) <= 0):
        raise ValueError("Frequencies must be positive and strictly increasing for ogive integration.")

    interval_integrals = 0.5 * (cospectrum[:-1] + cospectrum[1:]) * np.diff(frequency)
    return np.concatenate((np.cumsum(interval_integrals[::-1])[::-1], np.array([0.0])))


__all__ = ["compute_1d_spectrum", "compute_cospectrum", "compute_ogive"]
