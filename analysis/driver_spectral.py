
"""Energy-spectrum, cospectrum, and combined spectral-panel driver."""

from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from pathlib import Path

import numpy as np

from analysis.driver_common import align_height_series, artifact_from_figure, driver_parser, load_data, load_driver_config, resolve_flags
from analysis.models import PlotArtifact
from analysis.spectra import compute_1d_spectrum, compute_cospectrum
from tools.common import get_variable_time_axis
from tools.plotting import plot_cospectra, plot_energy_spectra, plot_spectral_panel
from tools.series import collect_height_series
from tools.style import apply_style

PLOT_SPECTRA = False
PLOT_COSPECTRA = False
PLOT_SPECTRAL_PANELS = False
SAVE_FIGURES = False

SPECTRUM_COMPONENTS = ("u", "v", "w", "tc")
COSPECTRUM_PAIRS = (("u", "w"), ("v", "w"), ("w", "tc"))
APPLY_LOG_BINNING = True
LOG_BINS_PER_DECADE = 12


def _spectra(data, config):
    _, time_local, _, _ = get_variable_time_axis(data)
    stats = []
    for prefix in SPECTRUM_COMPONENTS:
        output = []
        for entry in collect_height_series(data, config.site, prefix):
            frequency, spectrum = compute_1d_spectrum(
                entry["data"], entry["meta"], time_local,
                apply_log_binning=APPLY_LOG_BINNING, log_bins_per_decade=LOG_BINS_PER_DECADE,
            )
            output.append({**entry, "frequencyHz": frequency, "data": spectrum})
        stats.append({"varName": prefix, "displayName": prefix, "series": output})
    return stats


def _cospectra(data, config):
    _, time_local, _, _ = get_variable_time_axis(data)
    stats = []
    for prefix_x, prefix_y in COSPECTRUM_PAIRS:
        x_series = collect_height_series(data, config.site, prefix_x)
        y_series = collect_height_series(data, config.site, prefix_y)
        x_series, y_series = align_height_series(x_series, y_series)
        output = []
        for x_entry, y_entry in zip(x_series, y_series):
            frequency, values, covariance = compute_cospectrum(
                x_entry["data"], x_entry["meta"], y_entry["data"], y_entry["meta"], time_local,
                apply_log_binning=APPLY_LOG_BINNING, log_bins_per_decade=LOG_BINS_PER_DECADE,
            )
            normalized = np.abs(frequency * values) / max(abs(covariance), np.finfo(float).eps)
            output.append({**x_entry, "frequencyHz": frequency, "data": values, "normalizedData": normalized})
        stats.append({"varName": f"{prefix_x}_{prefix_y}", "displayName": f"Co({prefix_x}, {prefix_y})", "series": output})
    return stats


def run(config_path: str | Path | None = None, flag_overrides: dict[str, bool] | None = None) -> list[PlotArtifact]:
    flags = resolve_flags(
        {"plot_spectra": PLOT_SPECTRA, "plot_cospectra": PLOT_COSPECTRA,
         "plot_spectral_panels": PLOT_SPECTRAL_PANELS, "save_figures": SAVE_FIGURES},
        flag_overrides,
    )
    if not any(value for name, value in flags.items() if name.startswith("plot_")):
        return []
    config = load_driver_config(config_path)
    prefixes = list(SPECTRUM_COMPONENTS)
    for pair in COSPECTRUM_PAIRS:
        prefixes.extend(pair)
    data = load_data(config, list(dict.fromkeys(prefixes)))
    apply_style(config.figure)
    spectra = _spectra(data, config) if flags["plot_spectra"] or flags["plot_spectral_panels"] else []
    cospectra = _cospectra(data, config) if flags["plot_cospectra"] or flags["plot_spectral_panels"] else []
    artifacts = []
    if flags["plot_spectra"]:
        artifacts.append(artifact_from_figure(config, "spectra", plot_energy_spectra(spectra, f"{config.site.upper()}: spectra"), flags["save_figures"]))
    if flags["plot_cospectra"]:
        artifacts.append(artifact_from_figure(config, "cospectra", plot_cospectra(cospectra, f"{config.site.upper()}: cospectra"), flags["save_figures"]))
    if flags["plot_spectral_panels"]:
        artifacts.append(artifact_from_figure(config, "spectral_panels", plot_spectral_panel(spectra, cospectra, f"{config.site.upper()}: spectral panels"), flags["save_figures"]))
    return artifacts


def main() -> None:
    args = driver_parser(__doc__ or "Spectral analysis").parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
