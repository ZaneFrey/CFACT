from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from tools.plotting import plot_autocorrelation, plot_ogives


def test_autocorrelation_layout_and_variable_labels():
    lags = np.arange(4, dtype=float)
    stats = [
        {
            "varName": variable,
            "displayName": variable,
            "lagSeconds": lags,
            "series": [{"heightTag": "1m", "data": np.linspace(1.0, 0.0, lags.size)}],
        }
        for variable in ("u", "v", "w", "tc")
    ]

    figure = plot_autocorrelation(stats, "Autocorrelation")
    try:
        axes = figure.axes
        assert all(ax.get_legend() is None for ax in axes[:-1])
        assert axes[-1].get_legend() is not None
        assert [ax.get_ylabel() for ax in axes] == [
            r"$R_{uu}(L)$",
            r"$R_{vv}(L)$",
            r"$R_{ww}(L)$",
            r"$R_{T_cT_c}(L)$",
        ]
        assert np.isclose(figure.subplotpars.wspace, 0.28)
    finally:
        plt.close(figure)


def test_ogives_use_time_scale_on_the_x_axis():
    time_scale_seconds = np.array([10.0, 5.0, 2.5])
    figure = plot_ogives([{
        "varName": "u_w",
        "displayName": "Co(u, w)",
        "series": [{
            "heightTag": "1m",
            "timeScaleSeconds": time_scale_seconds,
            "data": np.array([0.35, 0.15, 0.0]),
        }],
    }])
    try:
        assert len(figure.axes) == 1
        assert figure.axes[0].get_ylabel().startswith("Ogive")
        assert figure.axes[0].get_xlabel() == "Time scale [s]"
        np.testing.assert_allclose(figure.axes[0].lines[0].get_xdata(), time_scale_seconds)
    finally:
        plt.close(figure)
