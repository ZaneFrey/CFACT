"""Wavelet-spectrum and scalogram driver."""

from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from pathlib import Path

from analysis.driver_common import artifact_from_figure, driver_parser, load_data, load_driver_config, resolve_flags
from analysis.models import PlotArtifact
from analysis.wavelets import compute_wavelet_analysis
from tools.common import get_variable_time_axis
from tools.plotting import plot_energy_spectra, plot_wavelet_scalogram
from tools.series import collect_height_series
from tools.style import apply_style

PLOT_WAVELET_SPECTRA = False
PLOT_SCALOGRAMS = True
SAVE_FIGURES = True

WAVELET_COMPONENT = "u"


def run(config_path: str | Path | None = None, flag_overrides: dict[str, bool] | None = None) -> list[PlotArtifact]:
    flags = resolve_flags(
        {"plot_wavelet_spectra": PLOT_WAVELET_SPECTRA, "plot_scalograms": PLOT_SCALOGRAMS, "save_figures": SAVE_FIGURES},
        flag_overrides,
    )
    if not flags["plot_wavelet_spectra"] and not flags["plot_scalograms"]:
        return []
    config = load_driver_config(config_path)
    data = load_data(config, [WAVELET_COMPONENT])
    _, time_local, _, _ = get_variable_time_axis(data)
    series = []
    for entry in collect_height_series(data, config.site, WAVELET_COMPONENT):
        frequency, energy, magnitude, native_time = compute_wavelet_analysis(entry["data"], entry["meta"], time_local)
        series.append({**entry, "frequencyHz": frequency, "data": energy, "scalogramMagnitude": magnitude, "scalogramTime": native_time})
    stats = {"varName": WAVELET_COMPONENT, "displayName": WAVELET_COMPONENT, "series": series}
    apply_style(config.figure)
    artifacts: list[PlotArtifact] = []
    if flags["plot_wavelet_spectra"]:
        figure = plot_energy_spectra([stats], f"{config.site.upper()}: wavelet spectrum")
        artifacts.append(artifact_from_figure(config, "wavelet_spectra", figure, flags["save_figures"]))
    if flags["plot_scalograms"]:
        figure = plot_wavelet_scalogram(stats, f"{config.site.upper()}: wavelet scalograms")
        artifacts.append(artifact_from_figure(config, "wavelet_scalograms", figure, flags["save_figures"]))
    return artifacts


def main() -> None:
    args = driver_parser(__doc__ or "Wavelet analysis").parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
