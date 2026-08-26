from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from tools.plotting import plot_autocorrelation


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
