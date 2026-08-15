from __future__ import annotations

from pathlib import Path

import pytest

import analysis.config as config_module
from analysis.config import load_config


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "base.yaml"


def _raw_config(**updates):
    raw = {
        "site": "dcs",
        "timezone": "America/Denver",
        "start_time": "2022-02-20 10:00:00",
        "end_time": "2022-02-20 12:00:00",
        "averaging_period_seconds": 300,
        "centered_gliding": True,
        "data_dir": "../../data",
        "output_dir": "../../analysis/outputs",
        "figure": {"format": "png", "dpi": 120, "width": 8, "panel_height": 2},
    }
    raw.update(updates)
    return raw


def test_config_resolves_relative_paths_and_localizes_time(monkeypatch):
    monkeypatch.setattr(config_module.yaml, "safe_load", lambda stream: _raw_config())
    config = load_config(FIXTURE_PATH)
    root = Path(__file__).parents[1]
    assert config.data_dir == (root / "data").resolve()
    assert config.output_dir == (root / "analysis" / "outputs").resolve()
    assert str(config.start_time.tz) == "America/Denver"
    assert config.averaging_period_seconds == 300


@pytest.mark.parametrize(
    ("updates", "error"),
    [
        ({"site": "invalid"}, ValueError),
        ({"timezone": "Mars/Olympus"}, ValueError),
        ({"end_time": "2022-02-20 09:00:00"}, ValueError),
        ({"averaging_period_seconds": 0}, ValueError),
        ({"data_dir": "missing"}, FileNotFoundError),
        ({"figure": {"format": "jpg"}}, ValueError),
    ],
)
def test_config_rejects_invalid_values(monkeypatch, updates, error):
    monkeypatch.setattr(config_module.yaml, "safe_load", lambda stream: _raw_config(**updates))
    with pytest.raises(error):
        load_config(FIXTURE_PATH)
