# Tools package

`tools` contains reusable infrastructure rather than scientific policy:

- `netcdf.py`: filtered high-rate/five-minute readers
- `selection.py`: timezone-aware filename/time overlap selection
- `series.py`: numeric height discovery, selection, and collection
- `plotting.py`: domain-neutral plotting layouts
- `style.py`: configuration-driven Matplotlib defaults
- `figures.py`: stable artifact naming and overwrite-aware saving
- `common.py`: site metadata, time-axis, height, and sanitization helpers

Public interfaces use snake_case. NetCDF inputs may be a file, a list of files, or a directory. The readers preserve native sample dimensions so high-rate statistics can be calculated from the original 20 Hz observations. Plot utilities sort heights numerically, map the lowest height to the darkest blue, place height legends outside axes, and explicitly label local time.

Tools must not decide which scientific products a driver enables. Likewise, scientific calculations belong in `analysis`, not in the NetCDF or figure-saving layers.
