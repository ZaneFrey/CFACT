from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.wavelets import compute_wavelet_analysis
from tools.plotting import plot_wavelet_scalogram


def test_wavelet_analysis_sanitizes_fill_values_before_computing_coefficients():
    time = pd.date_range("2022-02-20 10:00", periods=16, freq="1s", tz="America/Denver")
    values = np.sin(np.linspace(0, 2 * np.pi, time.size))
    values_with_fill = values.copy()
    values_with_fill[7] = -9999.0
    meta = {"attributes": {"_FillValue": -9999.0}}

    frequency, energy, magnitude, native_time = compute_wavelet_analysis(values_with_fill, meta, time)

    assert np.all(np.isfinite(frequency))
    assert np.all(np.isfinite(energy))
    assert np.all(np.isfinite(magnitude))
    assert native_time.equals(time)


def test_wavelet_scalogram_uses_local_time_and_shared_color_limits():
    time = pd.date_range("2022-02-20 10:00", periods=4, freq="1min", tz="America/Denver")
    stats = {
        "series": [
            {
                "heightTag": "2m",
                "frequencyHz": np.array([1.0, 0.5]),
                "scalogramMagnitude": np.array([[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]]),
                "scalogramTime": time,
            },
            {
                "heightTag": "10m",
                "frequencyHz": np.array([1.0, 0.5]),
                "scalogramMagnitude": np.array([[10.0, 11.0, 12.0, 13.0], [11.0, 12.0, 13.0, 14.0]]),
                "scalogramTime": time,
            },
        ]
    }

    figure = plot_wavelet_scalogram(stats)
    data_axes = [axis for axis in figure.axes if axis.get_ylabel() == "f [Hz]"]
    color_limits = [axis.collections[0].get_clim() for axis in data_axes]

    assert color_limits == [(1.0, 14.0), (1.0, 14.0)]
    assert all(isinstance(axis.xaxis.get_major_formatter(), mdates.DateFormatter) for axis in data_axes)
    assert all(axis.xaxis.get_major_formatter().tz == time.tz for axis in data_axes)
    plt.close(figure)
