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

## Driver controls and outputs

Every implemented driver exposes:

```python
run(config_path=None, flag_overrides=None) -> list[PlotArtifact]
```

Plot flags can be changed by editing the uppercase constants near the top of a driver or by passing snake_case Boolean overrides to `run()`. The command-line entry points accept `--config`; they do not expose individual plot flags.

When `save_figures` is `true`, each artifact is written to `output_dir` as `<artifact-name>.<figure.format>`. When it is `false`, the figure is still returned, but `saved_path` is `None`. If all supported plot flags are disabled, the driver returns an empty list without loading data.

Time-series calculations generally use `averaging_period_seconds` and `centered_gliding` from the configuration. The integral-timescale calculation is the exception described below: its scientific window is controlled by `MAX_LAG_SECONDS`, while the configuration controls only the plotted average. Plots use local time, sort heights numerically, assign the darkest blue to the lowest height, and place height legends outside the axes.

### `driver_metdata.py`

Produces window-mean meteorological time series. High-rate measurements are used except for radiation, which uses five-minute products.

| Override | Default | Plot and artifact output |
|---|---:|---|
| `plot_u` | `false` | Mean streamwise wind component, `u.<format>` |
| `plot_v` | `false` | Mean cross-stream wind component, `v.<format>` |
| `plot_w` | `false` | Mean vertical wind component, `w.<format>` |
| `plot_wind_speed` | `true` | Mean measured wind speed, `wind_speed.<format>` |
| `plot_wind_direction` | `true` | Meteorological direction derived from mean `u` and `v`, expressed as degrees from north, `wind_direction.<format>` |
| `plot_sonic_temperature` | `true` | Mean sonic temperature, `sonic_temperature.<format>` |
| `plot_ambient_temperature` | `false` | Mean ambient temperature, `ambient_temperature.<format>` |
| `plot_relative_humidity` | `true` | Mean relative humidity with a 0–110% display range, `relative_humidity.<format>` |
| `plot_radiation` | `false` | Four figures: `shortwave_incoming`, `shortwave_outgoing`, `longwave_incoming`, and `longwave_outgoing` |
| `save_figures` | `true` | Controls whether the returned figures are written to disk |

### `driver_stats.py`

Produces windowed velocity variances and raw-sample probability distributions.

| Override | Default | Plot and artifact output |
|---|---:|---|
| `plot_u_variance` | `true` | Windowed variance $\sigma_u^2$, `u_variance.<format>` |
| `plot_v_variance` | `true` | Windowed variance $\sigma_v^2$, `v_variance.<format>` |
| `plot_w_variance` | `true` | Windowed variance $\sigma_w^2$, `w_variance.<format>` |
| `plot_pdfs` | `false` | One `pdfs.<format>` figure containing PDF and CDF panels for `u`, `v`, `w`, and sonic temperature. PDF panels also report skewness and kurtosis by height. |
| `save_figures` | `true` | Controls figure persistence |

### `driver_correlations.py`

Produces autocorrelation, integral-timescale, and $u'$-$w'$ quadrant diagnostics.

| Override | Default | Plot and artifact output |
|---|---:|---|
| `plot_autocorrelation` | `false` | Autocorrelation panels for `u`, `v`, `w`, and sonic temperature, `autocorrelation.<format>` |
| `plot_integral_timescale` | `true` | Native-rate integral timescale for the configured component pair, averaged only for plotting, `integral_timescale.<format>` |
| `plot_quadrant_scatter` | `false` | Scatter plot of $u'$ versus $w'$, with ejection, sweep, inward-interaction, and outward-interaction quadrants, `quadrant_scatter.<format>` |
| `plot_quadrant_joint_pdf` | `false` | Joint-probability contours in the same quadrant coordinate system, `quadrant_joint_pdf.<format>` |
| `save_figures` | `true` | Controls figure persistence |

Additional source-level controls are:

- `AUTOCORRELATION_COMPONENTS`, defaulting to `("u", "v", "w", "tc")`.
- `MAX_LAG_SECONDS`, which sets both the maximum autocorrelation-plot lag and the centered calculation-window duration for the native-rate integral timescale.
- `INTEGRAL_TIMESCALE_PAIR`, defaulting to `("u", "u")`, so the standard output is an autocorrelation-based timescale.
- Integral timescales are calculated at every native-rate timestamp. `averaging_period_seconds` then controls only the displayed moving mean when `centered_gliding` is `true`, or nonoverlapping block means when it is `false`.
- The integral-timescale load interval includes `MAX_LAG_SECONDS / 2` of padding on each side. Native-rate results are cropped back to the configured interval; unavailable calculation padding produces `NaN` edge values.
- Quadrant fluctuations are computed by subtracting a moving mean whose width is `averaging_period_seconds`.

### `driver_spectral.py`

Produces Fourier energy spectra and cospectra from high-rate observations.

| Override | Default | Plot and artifact output |
|---|---:|---|
| `plot_spectra` | `true` | Energy-spectrum panels for `u`, `v`, `w`, and sonic temperature, including a -5/3 reference slope, `spectra.<format>` |
| `plot_cospectra` | `true` | Cospectrum panels for the configured component pairs. Each pair has normalized magnitude and signed $f\,Co(f)$ panels, `cospectra.<format>` |
| `plot_spectral_panels` | `false` | Intended to combine spectra and cospectra into one 3-by-4 figure, `spectral_panels.<format>` |
| `save_figures` | `true` | Controls figure persistence |

Additional source-level controls are:

- `SPECTRUM_COMPONENTS`, defaulting to `("u", "v", "w", "tc")`.
- `COSPECTRUM_PAIRS`, defaulting to `(("u", "v"), ("u", "w"), ("u", "tc"))`.
- `APPLY_LOG_BINNING`, defaulting to `true`.
- `LOG_BINS_PER_DECADE`, defaulting to 50.

### `driver_anisotropy.py`

Computes Reynolds-stress anisotropy in barycentric coordinates using the configured averaging window.

| Override | Default | Plot and artifact output |
|---|---:|---|
| `plot_x_b` | `true` | Barycentric $x_B$ time series by height, `anisotropy_x_b.<format>` |
| `plot_y_b` | `true` | Barycentric $y_B$ time series by height, `anisotropy_y_b.<format>` |
| `plot_anisotropy_spectra` | `false` | Energy spectra of $x_B$ and $y_B$, `anisotropy_spectra.<format>` |
| `plot_lumley_triangle` | `true` | All finite $(x_B,y_B)$ samples plotted in the Lumley/barycentric triangle and colored by height, `lumley_triangle.<format>` |
| `plot_trajectories` | `true` | Time-colored anisotropy trajectory at the height nearest `TRAJECTORY_HEIGHT_METERS`, `anisotropy_trajectory.<format>` |
| `plot_triangle_animation` | `false` | Unsupported placeholder; enabling it raises an actionable `NotImplementedError` |
| `save_figures` | `true` | Controls figure persistence |

`TRAJECTORY_HEIGHT_METERS` defaults to 17 m. The nearest available common `u`/`v`/`w` height is selected.

### `driver_tke.py`

Produces turbulent kinetic energy, friction velocity, Reynolds fluxes, moisture flux, and vertical TKE transport.

| Override | Default | Plot and artifact output |
|---|---:|---|
| `plot_tke` | `false` | $0.5(\sigma_u^2+\sigma_v^2+\sigma_w^2)$, `tke.<format>` |
| `plot_friction_velocity` | `false` | $u_*=(\overline{u'w'}^2+\overline{v'w'}^2)^{1/4}$, `friction_velocity.<format>` |
| `plot_reynolds_fluxes` | `false` | Three figures: `reynolds_flux_u_w`, `reynolds_flux_v_w`, and `reynolds_flux_w_w` |
| `plot_moisture_fluxes` | `false` | Specific-humidity flux derived from five-minute vapor, temperature, pressure, and vapor-flux products, `moisture_flux.<format>` |
| `plot_tke_transport` | `false` | Vertical turbulent transport $\overline{w'e}$, `tke_transport.<format>` |
| `plot_full_tke_budget` | `false` | Unsupported placeholder; enabling it raises `NotImplementedError` |
| `plot_z_over_l` | `true` | Unsupported placeholder; enabling it raises `NotImplementedError` |
| `save_figures` | `false` | Controls figure persistence |

Because `plot_z_over_l` currently defaults to `true`, an unmodified call to this driver raises the placeholder error. Set `plot_z_over_l` to `false` and enable one or more implemented flags to produce figures.

### `driver_wavelet.py`

Runs continuous wavelet analysis for one configured component at every available height.

| Override | Default | Plot and artifact output |
|---|---:|---|
| `plot_wavelet_spectra` | `false` | Height-colored wavelet energy spectra, `wavelet_spectra.<format>` |
| `plot_scalograms` | `true` | One time-frequency scalogram per height with logarithmic frequency axes and $\lvert WT\rvert$ color scales, `wavelet_scalograms.<format>` |
| `save_figures` | `true` | Controls figure persistence |

`WAVELET_COMPONENT` selects the analyzed variable and defaults to `"u"`.

### Placeholder drivers

`driver_mrd.py` and `driver_pod.py` retain the standard `run()` and `main()` interfaces but do not currently produce plots:

- `driver_mrd.py` always raises `NotImplementedError` until a validated multi-resolution decomposition is added.
- `driver_pod.py` always raises `NotImplementedError` until a validated snapshot or field decomposition is added.

### Known output limitations

- `plot_spectral_panels` currently cannot complete because the plotting function requires four cospectrum groups while `COSPECTRUM_PAIRS` defines three.
- The quadrant plots are labeled $u'/\sigma_u$ and $w'/\sigma_w$, but the driver currently supplies mean-removed fluctuations without dividing by their standard deviations.

## Default validation products

With the checked-in PNG configuration and current plot defaults, `driver_metdata.py` produces `wind_speed.png`, `wind_direction.png`, `sonic_temperature.png`, and `relative_humidity.png`. `driver_anisotropy.py` produces `anisotropy_x_b.png`, `anisotropy_y_b.png`, `lumley_triangle.png`, and `anisotropy_trajectory.png`. These products are written to `analysis/outputs/` and use the configured 900-second centered gliding calculations.
