from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

import analysis.driver_metdata as metdata


def test_plot_p_requests_only_the_dedicated_high_rate_pressure_prefix(monkeypatch):
    config = SimpleNamespace(site="dcs")
    loaded = object()
    calls = {}

    def fake_load_data(actual_config, prefixes, cadence="20hz"):
        calls["load"] = (actual_config, prefixes, cadence)
        return loaded

    pressure_series = [{"heightTag": "2m", "heightValue": 2.0, "data": [825.0]}]

    def fake_mean_height_series(data, actual_config, prefix):
        calls["mean"] = (data, actual_config, prefix)
        return pd.DatetimeIndex(["2022-02-20 10:00:00"], tz="America/Denver"), pressure_series

    def fake_plot_height_series(actual_config, name, title, ylabel, time_axis, series, **kwargs):
        calls["plot"] = (actual_config, name, title, ylabel, time_axis, series, kwargs)
        return "pressure-artifact"

    monkeypatch.setattr(metdata, "load_driver_config", lambda _: config)
    monkeypatch.setattr(metdata, "load_data", fake_load_data)
    monkeypatch.setattr(metdata, "mean_height_series", fake_mean_height_series)
    monkeypatch.setattr(metdata, "plot_height_series", fake_plot_height_series)

    artifacts = metdata.run(
        flag_overrides={
            "plot_u": False,
            "plot_v": False,
            "plot_w": False,
            "plot_wind_speed": False,
            "plot_wind_direction": False,
            "plot_sonic_temperature": False,
            "plot_ambient_temperature": False,
            "plot_relative_humidity": False,
            "plot_p": True,
            "save_figures": False,
        }
    )

    assert artifacts == ["pressure-artifact"]
    assert calls["load"] == (config, ["P"], "20hz")
    assert calls["mean"] == (loaded, config, "P")
    assert calls["plot"][1:4] == ("p", "DCS: Mean pressure", "Pressure [mb]")
    assert calls["plot"][5] is pressure_series
    assert calls["plot"][6] == {"y_limits": None, "save_figures": False}
