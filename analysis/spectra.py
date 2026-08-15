"""Fourier spectra and cospectra."""

from __future__ import annotations

from typing import Any

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


__all__ = ["compute_1d_spectrum", "compute_cospectrum"]
