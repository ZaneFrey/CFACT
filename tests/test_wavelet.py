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


def test_lumley_trajectory_uses_local_time_and_requested_visual_style():
    from tools.plotting import plot_lumley_trajectory

    time = pd.date_range("2022-02-20 10:00", periods=3, freq="1h", tz="America/Denver")
    series = [{"data": np.array([0.2, 0.5, 0.7])}]
    figure = plot_lumley_trajectory(series, [{"data": np.array([0.1, 0.3, 0.4])}], time)
    triangle_axis, colorbar_axis = figure.axes

    assert triangle_axis.axison is True
    assert triangle_axis.get_xlabel() == "$x_B$"
    assert triangle_axis.get_ylabel() == "$y_B$"
    assert all(not spine.get_visible() for spine in triangle_axis.spines.values())
    assert triangle_axis.collections[0].cmap.name == "seismic"
    assert colorbar_axis.get_ylabel() == "Local time"
    assert [label.get_text() for label in colorbar_axis.get_yticklabels()] == ["10:00", "11:00", "12:00"]
    figure.canvas.draw()
    triangle_bounds = triangle_axis.transData.transform([[0.0, 0.0], [1.0, np.sqrt(3.0) / 2.0]])
    colorbar_bounds = colorbar_axis.get_window_extent()
    assert np.isclose(colorbar_bounds.y0, triangle_bounds[0, 1])
    assert np.isclose(colorbar_bounds.y1, triangle_bounds[1, 1])
    plt.close(figure)
