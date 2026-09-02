"""Turbulent kinetic energy and simplified TKE-budget calculations."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from analysis._math import (
    compute_tke_transport_flux,
    compute_vertical_gradient_lagrange,
    expand_raw_series,
    moving_mean,
)
from tools.common import sanitize_series_array

GRAVITY_M_S2 = 9.80665
KOLMOGOROV_VELOCITY_CONSTANT = 0.55
# For a one-dimensional -5/3 velocity spectrum, the longitudinal
# second-order structure-function coefficient is 4.02 * alpha_k.
STRUCTURE_FUNCTION_SPECTRUM_FACTOR = 4.02


def turbulent_kinetic_energy(variance_u: Any, variance_v: Any, variance_w: Any) -> np.ndarray:
    return 0.5 * (
        np.asarray(variance_u, dtype=float)
        + np.asarray(variance_v, dtype=float)
        + np.asarray(variance_w, dtype=float)
    )


def friction_velocity(covariance_uw: Any, covariance_vw: Any) -> np.ndarray:
    uw = np.asarray(covariance_uw, dtype=float)
    vw = np.asarray(covariance_vw, dtype=float)
    return (uw**2 + vw**2) ** 0.25


def time_derivative_lagrange(time: Any, values: Any) -> np.ndarray:
    """Differentiate a time series with three-point Lagrange polynomials."""

    time_index = pd.DatetimeIndex(time)
    value_array = np.asarray(values, dtype=float).reshape(-1)
    if value_array.size != time_index.size:
        raise ValueError("The time axis and values must have the same length.")
    if value_array.size < 3:
        raise ValueError("At least 3 times are required to compute a time derivative.")
    ticks_ns = time_index.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    seconds = (ticks_ns - ticks_ns[0]).astype(float) / 1e9
    if np.any(np.diff(seconds) <= 0):
        raise ValueError("The time axis must be strictly increasing.")
    return compute_vertical_gradient_lagrange(seconds, value_array)


def buoyancy_production(
    mean_virtual_temperature_k: Any,
    vertical_virtual_temperature_flux: Any,
    gravity_m_s2: float = GRAVITY_M_S2,
) -> np.ndarray:
    """Return ``g / mean(theta_v) * covariance(w, theta_v)``.

    A pressure conversion that is common within an averaging window cancels
    from this ratio, so virtual sonic temperature can be used directly.
    """

    temperature, flux = np.broadcast_arrays(
        np.asarray(mean_virtual_temperature_k, dtype=float),
        np.asarray(vertical_virtual_temperature_flux, dtype=float),
    )
    out = np.full(temperature.shape, np.nan)
    valid = np.isfinite(temperature) & np.isfinite(flux) & (temperature > 0)
    out[valid] = float(gravity_m_s2) * flux[valid] / temperature[valid]
    return out


def dissipation_from_structure_parameter(
    structure_parameter_uu: Any,
    alpha_k: float = KOLMOGOROV_VELOCITY_CONSTANT,
) -> np.ndarray:
    """Convert longitudinal ``C_uu`` to dissipation using the 2/3 law."""

    if not np.isfinite(alpha_k) or alpha_k <= 0:
        raise ValueError("alpha_k must be a positive finite value.")
    c_uu = np.asarray(structure_parameter_uu, dtype=float)
    out = np.full(c_uu.shape, np.nan)
    valid = np.isfinite(c_uu) & (c_uu >= 0)
    out[valid] = (c_uu[valid] / (STRUCTURE_FUNCTION_SPECTRUM_FACTOR * alpha_k)) ** 1.5
    return out


def _structure_parameter_from_increment(
    increment_squared: np.ndarray,
    mean_streamwise_speed: np.ndarray,
    sample_interval_seconds: float,
    separation_samples: int,
) -> np.ndarray:
    separation = (
        np.abs(np.asarray(mean_streamwise_speed, dtype=float))
        * float(sample_interval_seconds)
        * int(separation_samples)
    )
    out = np.full(np.broadcast_shapes(increment_squared.shape, separation.shape), np.nan)
    increments, separation = np.broadcast_arrays(increment_squared, separation)
    valid = np.isfinite(increments) & np.isfinite(separation) & (increments >= 0) & (separation > 0)
    out[valid] = increments[valid] / separation[valid] ** (2.0 / 3.0)
    return out


def compute_dissipation_rate(
    streamwise_velocity: Any,
    meta: dict[str, Any] | None,
    time_second: Any,
    averaging_period_seconds: float = 1.0,
    glide: bool = True,
    *,
    alpha_k: float = KOLMOGOROV_VELOCITY_CONSTANT,
    separation_samples: int = 1,
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Estimate epsilon from the longitudinal second-order structure function.

    Taylor's frozen-turbulence hypothesis converts the temporal separation to
    ``r = abs(mean_u) * lag``. ``C_uu = D_uu(r) / r**(2/3)`` is then converted
    using ``D_uu = 4.02 * alpha_k * epsilon**(2/3) * r**(2/3)``.
    """

    if int(separation_samples) != separation_samples or separation_samples < 1:
        raise ValueError("separation_samples must be a positive integer.")
    raw, representative_indices, sample_rate_hz, raw_time = expand_raw_series(
        streamwise_velocity, meta, time_second
    )
    raw = sanitize_series_array(raw, meta).reshape(-1)
    window_samples = max(1, int(round(float(averaging_period_seconds) * sample_rate_hz)))
    lag = int(separation_samples)
    increment_squared = np.full(raw.size, np.nan)
    increment_squared[lag:] = (raw[lag:] - raw[:-lag]) ** 2
    sample_interval = 1.0 / sample_rate_hz

    if glide:
        mean_u = moving_mean(raw, window_samples)
        mean_increment_squared = moving_mean(increment_squared, window_samples)
        c_uu = _structure_parameter_from_increment(
            mean_increment_squared, mean_u, sample_interval, lag
        )
        return pd.DatetimeIndex(time_second), dissipation_from_structure_parameter(c_uu, alpha_k)[
            representative_indices
        ]

    n_block = int(math.ceil(raw.size / window_samples))
    epsilon = np.full(n_block, np.nan)
    output_time: list[pd.Timestamp] = []
    for index in range(n_block):
        start = index * window_samples
        stop = min((index + 1) * window_samples, raw.size)
        block = raw[start:stop]
        if block.size > lag:
            d_uu = float(np.nanmean((block[lag:] - block[:-lag]) ** 2))
            mean_u = float(np.nanmean(block))
            c_uu = _structure_parameter_from_increment(
                np.asarray(d_uu), np.asarray(mean_u), sample_interval, lag
            )
            epsilon[index] = float(dissipation_from_structure_parameter(c_uu, alpha_k))
        output_time.append(raw_time[start] + (raw_time[stop - 1] - raw_time[start]) / 2)
    return pd.DatetimeIndex(output_time), epsilon


def simplified_tke_budget(
    time: Any,
    heights_m: Any,
    tke: Any,
    mean_u: Any,
    mean_w: Any,
    covariance_uw: Any,
    mean_virtual_temperature_k: Any,
    vertical_virtual_temperature_flux: Any,
    vertical_tke_flux: Any,
    dissipation_rate: Any,
) -> dict[str, np.ndarray]:
    """Calculate terms 1--5 and 7 plus the pressure-inclusive residual.

    Every height-dependent input must have shape ``(height, time)``. The
    returned dissipation term carries its budget sign and is therefore
    ``-epsilon``.
    """

    heights = np.asarray(heights_m, dtype=float).reshape(-1)
    arrays = {
        "tke": np.asarray(tke, dtype=float),
        "mean_u": np.asarray(mean_u, dtype=float),
        "mean_w": np.asarray(mean_w, dtype=float),
        "covariance_uw": np.asarray(covariance_uw, dtype=float),
        "mean_virtual_temperature_k": np.asarray(mean_virtual_temperature_k, dtype=float),
        "vertical_virtual_temperature_flux": np.asarray(vertical_virtual_temperature_flux, dtype=float),
        "vertical_tke_flux": np.asarray(vertical_tke_flux, dtype=float),
        "dissipation_rate": np.asarray(dissipation_rate, dtype=float),
    }
    expected_shape = (heights.size, len(pd.DatetimeIndex(time)))
    for name, values in arrays.items():
        if values.shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}, got {values.shape}.")

    storage = np.vstack([time_derivative_lagrange(time, row) for row in arrays["tke"]])
    gradient_tke = compute_vertical_gradient_lagrange(heights, arrays["tke"])
    gradient_mean_u = compute_vertical_gradient_lagrange(heights, arrays["mean_u"])
    gradient_tke_flux = compute_vertical_gradient_lagrange(heights, arrays["vertical_tke_flux"])
    advection = -arrays["mean_w"] * gradient_tke
    buoyancy = buoyancy_production(
        arrays["mean_virtual_temperature_k"], arrays["vertical_virtual_temperature_flux"]
    )
    shear = -arrays["covariance_uw"] * gradient_mean_u
    transport = -gradient_tke_flux
    dissipation = -arrays["dissipation_rate"]
    residual = storage - (advection + buoyancy + shear + transport + dissipation)
    return {
        "storage": storage,
        "tke_advection": advection,
        "buoyancy_production": buoyancy,
        "shear_production": shear,
        "tke_transport": transport,
        "dissipation": dissipation,
        "residual": residual,
    }


__all__ = [
    "compute_tke_transport_flux",
    "compute_vertical_gradient_lagrange",
    "buoyancy_production",
    "compute_dissipation_rate",
    "dissipation_from_structure_parameter",
    "friction_velocity",
    "simplified_tke_budget",
    "time_derivative_lagrange",
    "turbulent_kinetic_energy",
]
