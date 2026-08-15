# Repository guidance

- Maintain a Python-only repository. Do not migrate MATLAB code, parity scripts, or standalone overview utilities.
- Keep all public names and driver flag overrides in snake_case; do not add `CFACT_*` compatibility wrappers.
- Keep drivers as thin orchestration layers. Scientific calculations belong in domain modules under `analysis`; I/O, time/file selection, height collection, plotting, style, and saving belong under `tools`.
- Every working driver must retain both `run(config_path=None, flag_overrides=None) -> list[PlotArtifact]` and `main()` and remain directly runnable from the repository root.
- Unsupported analyses must raise an actionable error when enabled. Current documented placeholders are MRD, POD, full TKE budget, z/L, and triangle animation.
- Never stage or commit NetCDF files or generated figures. Leave implementation changes unstaged unless the user explicitly asks otherwise.
- Preserve the visual convention that height order is numeric, the lowest height is darkest blue, legends are outside axes, and local time and x_B/y_B notation are explicit.
