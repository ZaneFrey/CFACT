from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.tke import (
    GRAVITY_M_S2,
    STRUCTURE_FUNCTION_SPECTRUM_FACTOR,
    dissipation_from_structure_parameter,
    simplified_tke_budget,
    time_derivative_lagrange,
)


def test_time_derivative_lagrange_is_exact_for_a_quadratic():
    seconds = np.asarray([0.0, 1.0, 3.0, 6.0])
    time = pd.DatetimeIndex(pd.Timestamp("2022-01-01", tz="UTC") + pd.to_timedelta(seconds, unit="s"))

    derivative = time_derivative_lagrange(time, seconds**2)

    np.testing.assert_allclose(derivative, 2.0 * seconds)


def test_dissipation_conversion_inverts_the_two_thirds_law():
    epsilon = np.asarray([0.01, 0.1, 1.0])
    alpha_k = 0.55
    c_uu = STRUCTURE_FUNCTION_SPECTRUM_FACTOR * alpha_k * epsilon ** (2.0 / 3.0)

    np.testing.assert_allclose(dissipation_from_structure_parameter(c_uu, alpha_k), epsilon)


def test_simplified_budget_uses_expected_signs_and_closes_with_residual():
    time = pd.date_range("2022-01-01", periods=4, freq="1s", tz="UTC")
    heights = np.asarray([1.0, 2.0, 4.0])
    elapsed = np.arange(time.size, dtype=float)
    z = heights[:, None]
    t = elapsed[None, :]
    tke = z**2 + t
    mean_u = np.broadcast_to(3.0 * z, tke.shape)
    mean_w = np.full(tke.shape, 2.0)
    covariance_uw = np.full(tke.shape, 0.5)
    mean_temperature = np.full(tke.shape, 300.0)
    temperature_flux = np.full(tke.shape, 0.3)
    tke_flux = np.broadcast_to(z**2, tke.shape)
    epsilon = np.full(tke.shape, 0.2)

    terms = simplified_tke_budget(
        time,
        heights,
        tke,
        mean_u,
        mean_w,
        covariance_uw,
        mean_temperature,
        temperature_flux,
        tke_flux,
        epsilon,
    )

    expected_gradient = np.broadcast_to(2.0 * z, tke.shape)
    np.testing.assert_allclose(terms["storage"], 1.0)
    np.testing.assert_allclose(terms["tke_advection"], -2.0 * expected_gradient)
    np.testing.assert_allclose(terms["buoyancy_production"], GRAVITY_M_S2 * 0.3 / 300.0)
    np.testing.assert_allclose(terms["shear_production"], -1.5)
    np.testing.assert_allclose(terms["tke_transport"], -expected_gradient)
    np.testing.assert_allclose(terms["dissipation"], -0.2)
    known_rhs = sum(terms[name] for name in terms if name not in {"storage", "residual"})
    np.testing.assert_allclose(known_rhs + terms["residual"], terms["storage"])
