# Analysis guidance

- Drivers define plot flags and plot-specific layouts, limits, component pairs, and spectral settings.
- Domain modules implement calculations and expose snake_case interfaces independently of plotting.
- Do not hide unsupported behavior behind warnings or empty results; raise an actionable exception.
- Retain partial centered windows at configured analysis boundaries.
- A driver `run` result must contain stable `PlotArtifact` names and must only save when `save_figures` is true.
