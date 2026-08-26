from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

import analysis.driver_correlations as correlations


def test_integral_timescale_uses_max_lag_then_plot_only_averaging(monkeypatch):
    config = SimpleNamespace(
        site="dcs",
        start_time=pd.Timestamp("2022-01-01 10:00:00", tz="America/Denver"),
        end_time=pd.Timestamp("2022-01-01 10:00:02", tz="America/Denver"),
        averaging_period_seconds=30.0,
        centered_gliding=False,
        figure=SimpleNamespace(),
    )
    padded_second_time = pd.date_range("2022-01-01 09:59:58", periods=7, freq="1s", tz="America/Denver")
    raw_time = pd.date_range("2022-01-01 09:59:59", periods=9, freq="500ms", tz="America/Denver")
    raw_values = np.arange(raw_time.size, dtype=float)
    entry = {
        "heightTag": "1m",
        "heightValue": 1.0,
        "varName": "u_1m_dcs",
        "data": np.arange(padded_second_time.size, dtype=float),
        "meta": None,
    }
    artifact = object()
    calls = {}

    monkeypatch.setattr(correlations, "MAX_LAG_SECONDS", 4.0)
    monkeypatch.setattr(correlations, "load_driver_config", lambda path: config)

    def fake_load_data(received_config, prefixes, cadence="20hz", time_padding_seconds=0.0):
        calls["padding"] = time_padding_seconds
        calls["prefixes"] = prefixes
        return {"loaded": True}

    monkeypatch.setattr(correlations, "load_data", fake_load_data)
    monkeypatch.setattr(
        correlations,
        "get_variable_time_axis",
        lambda data: (padded_second_time.tz_convert("UTC"), padded_second_time, "time_datetime", "time_datetime_local"),
    )
    monkeypatch.setattr(correlations, "collect_height_series", lambda *args, **kwargs: [entry])
    monkeypatch.setattr(correlations, "apply_style", lambda *args, **kwargs: None)

    def fake_integral_timescale(x, meta_x, y, meta_y, t_second, max_lag_seconds):
        calls["max_lag"] = max_lag_seconds
        return raw_time, raw_values

    monkeypatch.setattr(correlations, "compute_integral_timescale", fake_integral_timescale)

    def fake_window_stat(values, meta, time, averaging_period_seconds, centered_gliding, stat_name):
        calls["smoothed_values"] = np.asarray(values)
        calls["smoothed_time"] = pd.DatetimeIndex(time)
        calls["averaging_period"] = averaging_period_seconds
        calls["centered_gliding"] = centered_gliding
        calls["stat_name"] = stat_name
        return pd.DatetimeIndex(time), np.asarray(values)

    monkeypatch.setattr(correlations, "compute_window_stat", fake_window_stat)
    monkeypatch.setattr(correlations, "plot_height_series", lambda *args, **kwargs: artifact)

    result = correlations.run(
        flag_overrides={
            "plot_autocorrelation": False,
            "plot_integral_timescale": True,
            "plot_quadrant_scatter": False,
            "plot_quadrant_joint_pdf": False,
            "save_figures": False,
        }
    )

    assert result == [artifact]
    assert calls["padding"] == 2.0
    assert calls["prefixes"] == ["u"]
    assert calls["max_lag"] == 4.0
    assert calls["averaging_period"] == 30.0
    assert calls["centered_gliding"] is False
    assert calls["stat_name"] == "mean"
    assert calls["smoothed_time"].equals(raw_time[2:7])
    np.testing.assert_array_equal(calls["smoothed_values"], raw_values[2:7])
