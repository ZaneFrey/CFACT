# Local CFACT data

Download the NCAR/EOL CFACT ISFS products from the project data catalog and place the `.nc` files directly in this directory. Supported products are:

- high-rate surface meteorology and flux files: `isfs_cfact_hr_*_YYYYMMDD_HH.nc`
- five-minute surface meteorology and flux files: `isfs_cfact_5min_*_YYYYMMDD.nc`

Keep the directory flat; time-aware selection parses the UTC date/hour suffix from each filename and converts requested local bounds using the configured IANA time zone.

All NetCDF files below `data/` are ignored by Git. Do not commit raw campaign observations. Tests that require them skip when no `.nc` files are installed.
