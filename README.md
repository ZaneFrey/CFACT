# CFACT Python Analysis

This repository contains the Python analysis workflow for the Cold Fog Amongst Complex Terrain (CFACT) campaign. It reads NCAR/EOL ISFS high-rate and five-minute surface meteorology and flux NetCDF products, computes turbulence and meteorological diagnostics, and produces publication-ready figures.

The repository is intentionally Python-only. MATLAB sources, parity utilities, and the old standalone overview utility are not part of this codebase.

## Setup

Create or update the reproducible Conda environment:

```powershell
conda env create -f environment.yml
conda activate cfact
```

For an existing environment, use `conda env update -n cfact -f environment.yml --prune`.

## Data

Place NetCDF files directly in `data/`. Both supported filename forms are recognized:

- High rate: `isfs_cfact_hr_*_YYYYMMDD_HH.nc`
- Five minute: `isfs_cfact_5min_*_YYYYMMDD.nc`

Raw data and generated figures are ignored by Git. See [data/README.md](data/README.md) for acquisition and layout details.

## Run an analysis

The ready-to-run case in `analysis/config.yaml` selects the DCS site from 10:00 through 12:00 local time on 2022-02-20 and uses centered, partial-edge, 300-second gliding calculations.

```powershell
conda run -n cfact python analysis/driver_metdata.py
conda run -n cfact python analysis/driver_anisotropy.py --config analysis/config.yaml
```

Every working driver exposes `run(config_path=None, flag_overrides=None) -> list[PlotArtifact]` for programmatic use. Plot flags and plot-specific settings are constants near the top of each driver. `SAVE_FIGURES` controls persistence; saved files go to the configured output directory.

```python
from analysis.driver_metdata import run

artifacts = run(flag_overrides={
    "plot_wind_speed": True,
    "plot_wind_direction": False,
    "plot_sonic_temperature": False,
    "save_figures": True,
})
print(artifacts[0].name, artifacts[0].saved_path)
```

Available drivers are organized by domain:

- `driver_metdata.py`: u/v/w, wind speed/direction, sonic/ambient temperature, RH, radiation flag
- `driver_stats.py`: component variances and PDFs/histograms
- `driver_correlations.py`: autocorrelation, integral timescale, quadrant scatter, joint PDF
- `driver_spectral.py`: spectra, cospectra, combined panels
- `driver_tke.py`: TKE, friction velocity, Reynolds fluxes, supported TKE transport
- `driver_anisotropy.py`: x_B/y_B time series and explicit advanced-analysis flags
- `driver_wavelet.py`: wavelet spectra and scalograms
- `driver_mrd.py`, `driver_pod.py`: actionable unsupported placeholders

MRD, POD, the full TKE budget, z/L, and triangle animation remain explicit placeholders. Enabling unsupported flags raises `NotImplementedError`; it never silently produces no output.

## Validation

```powershell
conda run -n cfact pytest
conda run -n cfact python -m compileall -q analysis tools tests
git diff --check
git status --short
```

Integration tests skip cleanly when raw data are absent. With the validation data installed, they assert the six DCS heights (1, 2, 3, 7, 17, and 32 m) and validate the five default PNG products.

## AI disclaimer

Parts of this analysis framework were generated and streamlined with OpenAI models and require the same scientific review as any other contributed code.
