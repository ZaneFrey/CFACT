from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.anisotropy import barycentric_coordinates
from analysis.driver_common import wind_direction
from analysis.statistics import compute_window_stat


def test_centered_gliding_mean_retains_partial_edges():
    time = pd.date_range("2022-01-01", periods=5, freq="1s", tz="UTC")
    output_time, values = compute_window_stat(np.arange(1.0, 6.0), None, time, 3, True, "mean")
    assert output_time.equals(time)
    np.testing.assert_allclose(values, [1.5, 2.0, 3.0, 4.0, 4.5])


def test_wind_direction_cardinal_points():
    result = wind_direction([0, -1, 0, 1], [-1, 0, 1, 0])
    np.testing.assert_allclose(result, [0, 90, 180, 270])
    assert np.all((result >= 0) & (result < 360))


def test_barycentric_coordinates_stay_in_physical_bounds():
    rng = np.random.default_rng(42)
    time = pd.date_range("2022-01-01", periods=800, freq="50ms", tz="UTC")
    u = rng.normal(size=time.size)
    v = 0.7 * u + rng.normal(scale=0.5, size=time.size)
    w = rng.normal(scale=0.3, size=time.size)
    _, x_b, y_b = barycentric_coordinates(u, None, v, None, w, None, time, 5, True)
    finite = np.isfinite(x_b) & np.isfinite(y_b)
    assert finite.any()
    assert np.all((x_b[finite] >= 0) & (x_b[finite] <= 1))
    assert np.all((y_b[finite] >= 0) & (y_b[finite] <= np.sqrt(3) / 2))
