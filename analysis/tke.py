"""Turbulent kinetic energy, flux, and supported transport calculations."""

from __future__ import annotations

from typing import Any

import numpy as np

from analysis._math import compute_tke_transport_flux, compute_vertical_gradient_lagrange


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


__all__ = [
    "compute_tke_transport_flux",
    "compute_vertical_gradient_lagrange",
    "friction_velocity",
    "turbulent_kinetic_energy",
]
