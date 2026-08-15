# Analysis package

`analysis` owns scientific calculations and runnable domain drivers. It is importable as a package, while each `driver_*.py` can also be executed directly from the repository root.

## Configuration schema

`config.yaml` contains:

- `site`: NCAR site suffix such as `dcs` or `prs`
- `timezone`: IANA zone used for selection and labels
- `start_time`, `end_time`: local analysis bounds
- `averaging_period_seconds`: positive window duration
- `centered_gliding`: `true` for centered rolling windows with partial edge windows
- `data_dir`, `output_dir`: paths resolved relative to the YAML file
- `figure`: format (`png`, `pdf`, or `svg`), overwrite, DPI, size, font sizes, line width, grid, and height colormap settings

The loader rejects unsupported sites and formats, invalid or reversed times, nonpositive periods/sizes, invalid zones, and missing data directories. It does not create the output directory until a driver saves a figure.

## Driver workflow

Edit top-level `PLOT_*` constants for a repeatable script workflow, or pass snake_case overrides to `run`:

```python
from analysis.driver_stats import run

figures = run(
    "analysis/config.yaml",
    {"plot_u_variance": True, "plot_pdfs": True, "save_figures": False},
)
```

Each result is a `PlotArtifact` with a stable `name`, Matplotlib `figure`, and `saved_path` (or `None` if saving is disabled). Unknown or non-boolean overrides fail validation.

Computations are grouped into `statistics.py`, `spectra.py`, `anisotropy.py`, `thermodynamics.py`, `tke.py`, and `wavelets.py`. `_math.py` contains shared numerical kernels and is not a compatibility API.

## Default validation products

The default metdata and anisotropy flags produce separate `wind_speed.png`, `wind_direction.png`, `sonic_temperature.png`, `anisotropy_x_b.png`, and `anisotropy_y_b.png` files in `analysis/outputs/`. All are derived from high-rate observations with 300-second centered gliding calculations.
