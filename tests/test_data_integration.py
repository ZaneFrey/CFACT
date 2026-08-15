from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from analysis.driver_anisotropy import run as run_anisotropy
from analysis.driver_metdata import run as run_metdata


CONFIG = Path(__file__).parents[1] / "analysis" / "config.yaml"
DATA_DIR = Path(__file__).parents[1] / "data"
EXPECTED_HEIGHTS = ["1 m", "2 m", "3 m", "7 m", "17 m", "32 m"]


pytestmark = pytest.mark.skipif(not any(DATA_DIR.glob("*.nc")), reason="CFACT raw data are not installed")


def _validate_artifact(artifact, bounds=None):
    assert artifact.saved_path is not None
    assert artifact.saved_path.is_file()
    assert artifact.saved_path.stat().st_size > 0
    axes = artifact.figure.axes
    assert axes
    plotted = axes[0].lines
    assert len(plotted) == 6
    assert [line.get_label() for line in plotted] == EXPECTED_HEIGHTS
    values = np.concatenate([np.asarray(line.get_ydata(), dtype=float) for line in plotted])
    assert np.isfinite(values).any()
    if bounds is not None:
        finite = values[np.isfinite(values)]
        assert np.all((finite >= bounds[0]) & (finite <= bounds[1]))


def test_validation_case_generates_five_physical_six_height_products():
    met = run_metdata(CONFIG)
    aniso = run_anisotropy(CONFIG)
    artifacts = {artifact.name: artifact for artifact in met + aniso}
    assert set(artifacts) == {
        "wind_speed",
        "wind_direction",
        "sonic_temperature",
        "anisotropy_x_b",
        "anisotropy_y_b",
    }
    _validate_artifact(artifacts["wind_speed"])
    _validate_artifact(artifacts["wind_direction"], (0, 360))
    _validate_artifact(artifacts["sonic_temperature"])
    _validate_artifact(artifacts["anisotropy_x_b"], (0, 1))
    _validate_artifact(artifacts["anisotropy_y_b"], (0, np.sqrt(3) / 2))
