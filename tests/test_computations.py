from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.anisotropy import barycentric_coordinates
from analysis.driver_common import wind_direction
from analysis.statistics import compute_integral_timescale, compute_window_stat
from analysis.spectra import compute_ogive


def test_centered_gliding_mean_retains_partial_edges():
    time = pd.date_range("2022-01-01", periods=5, freq="1s", tz="UTC")
    output_time, values = compute_window_stat(np.arange(1.0, 6.0), None, time, 3, True, "mean")
    assert output_time.equals(time)
    np.testing.assert_allclose(values, [1.5, 2.0, 3.0, 4.0, 4.5])


def test_block_mean_uses_nonoverlapping_native_rate_windows():
    time = pd.date_range("2022-01-01", periods=8, freq="500ms", tz="UTC")
    output_time, values = compute_window_stat(np.arange(1.0, 9.0), None, time, 2, False, "mean")
    assert output_time.equals(pd.DatetimeIndex([time[0] + pd.Timedelta(milliseconds=750), time[4] + pd.Timedelta(milliseconds=750)]))
    np.testing.assert_allclose(values, [2.5, 6.5])


def test_integral_timescale_is_computed_at_native_rate_with_nan_edges():
    time = pd.date_range("2022-01-01", periods=6, freq="1s", tz="UTC")
    raw = np.resize(np.array([1.0, -1.0]), 12)
    values = raw.reshape((2, 6), order="F")
    meta = {"dimensions": ["sample", "time"], "originalName": "u"}

    output_time, timescale = compute_integral_timescale(values, meta, values, meta, time, 2.0)

    assert len(output_time) == raw.size
    assert len(timescale) == raw.size
    np.testing.assert_allclose(np.diff(output_time.asi8) / 1e9, 0.5)
    assert output_time[0] == time[0] - pd.Timedelta(milliseconds=250)
    assert np.isnan(timescale[:2]).all()
    assert np.isnan(timescale[-2:]).all()
    np.testing.assert_allclose(timescale[2:-2], 0.125)


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


def test_ogive_integrates_cospectrum_from_high_to_low_frequency():
    ogive = compute_ogive([1.0, 2.0, 4.0], [2.0, 2.0, 2.0])

    np.testing.assert_allclose(ogive, [6.0, 4.0, 0.0])
