"""Thermodynamic conversions used by meteorological analyses."""

from __future__ import annotations

from typing import Any

import numpy as np

from analysis._math import compute_specific_humidity_flux as _compute_specific_humidity_flux


def specific_humidity_flux(
    vapor_density_g_m3: Any,
    temperature_c: Any,
    pressure_mb: Any,
    vertical_vapor_density_flux_g_m3_m_s: Any,
) -> tuple[np.ndarray, np.ndarray]:
    return _compute_specific_humidity_flux(
        vapor_density_g_m3,
        temperature_c,
        pressure_mb,
        vertical_vapor_density_flux_g_m3_m_s,
    )


def potential_temperature(temperature_c: Any, pressure_mb: Any) -> np.ndarray:
    temperature_k = np.asarray(temperature_c, dtype=float) + 273.15
    pressure = np.asarray(pressure_mb, dtype=float)
    out = np.full(np.broadcast_shapes(temperature_k.shape, pressure.shape), np.nan)
    temperature_k, pressure = np.broadcast_arrays(temperature_k, pressure)
    valid = np.isfinite(temperature_k) & np.isfinite(pressure) & (temperature_k > 0) & (pressure > 0)
    out[valid] = temperature_k[valid] * (1000.0 / pressure[valid]) ** 0.2854
    return out


def virtual_potential_temperature(temperature_c: Any, pressure_mb: Any, specific_humidity: Any) -> np.ndarray:
    theta, q = np.broadcast_arrays(potential_temperature(temperature_c, pressure_mb), np.asarray(specific_humidity, dtype=float))
    out = np.full(theta.shape, np.nan)
    valid = np.isfinite(theta) & np.isfinite(q)
    out[valid] = theta[valid] * (1.0 + 0.61 * q[valid])
    return out


def relative_humidity_over_ice(relative_humidity_liquid: Any, temperature_c: Any) -> np.ndarray:
    rh, temperature = np.broadcast_arrays(
        np.asarray(relative_humidity_liquid, dtype=float), np.asarray(temperature_c, dtype=float)
    )
    liquid = 611.2 * np.exp(17.67 * temperature / (temperature + 243.5))
    ice = 611.2 * np.exp(22.46 * temperature / (temperature + 272.62))
    out = np.full(rh.shape, np.nan)
    valid = np.isfinite(rh) & np.isfinite(liquid) & np.isfinite(ice) & (ice > 0)
    out[valid] = rh[valid] * liquid[valid] / ice[valid]
    return out


__all__ = [
    "potential_temperature",
    "relative_humidity_over_ice",
    "specific_humidity_flux",
    "virtual_potential_temperature",
]
