from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from tools.common import convert_to_local_time, normalize_nc_inputs, normalize_time_bound


def _parse_file_window(nc_file: str) -> tuple[pd.Timestamp | None, pd.Timestamp | None, str, float | None, bool]:
    path = Path(nc_file)
    name = path.stem
    if path.suffix.lower() != ".nc":
        return None, None, "unknown", None, False
    if "_5min_" in name.lower():
        match = re.search(r"_(\d{8})$", name)
        if match:
            start = pd.to_datetime(match.group(1), format="%Y%m%d", utc=True)
            return start, start + pd.Timedelta(days=1) - pd.Timedelta(seconds=1), "5min", 300.0, True
    match = re.search(r"_(\d{8})_(\d{2})$", name)
    if match:
        start = pd.to_datetime(f"{match.group(1)}{match.group(2)}", format="%Y%m%d%H", utc=True)
        return start, start + pd.Timedelta(hours=1) - pd.Timedelta(seconds=1), "20hz", 1.0, True
    match = re.search(r"_(\d{8})$", name)
    if match:
        start = pd.to_datetime(match.group(1), format="%Y%m%d", utc=True)
        return start, start + pd.Timedelta(days=1) - pd.Timedelta(seconds=1), "unknown", None, True
    return None, None, "unknown", None, False


def select_files_by_local_timerange(nc_input: Any, start_time_local: Any, end_time_local: Any, local_timezone: str = "America/Denver"):
    files = normalize_nc_inputs(nc_input, script_dir=Path(__file__).resolve().parent.parent)
    start_local = normalize_time_bound(start_time_local, local_timezone)
    end_local = normalize_time_bound(end_time_local, local_timezone)
    if start_local is not None and end_local is not None and end_local < start_local:
        raise ValueError("end_time_local must be later than or equal to start_time_local.")
    start_utc = start_local.tz_convert("UTC") if start_local is not None else None
    end_utc = end_local.tz_convert("UTC") if end_local is not None else None
    rows = []
    selected = []
    for nc_file in files:
        file_start_utc, file_end_utc, dataset_type, cadence_seconds, parsed = _parse_file_window(nc_file)
        include = True
        if parsed and file_start_utc is not None and file_end_utc is not None:
            if start_utc is not None:
                include &= file_end_utc >= start_utc
            if end_utc is not None:
                include &= file_start_utc <= end_utc
        if include:
            selected.append(nc_file)
        rows.append(
            {
                "File": nc_file,
                "StartTimeUTC": file_start_utc,
                "EndTimeUTC": file_end_utc,
                "StartTimeLocal": convert_to_local_time([file_start_utc], local_timezone)[0] if file_start_utc is not None else pd.NaT,
                "EndTimeLocal": convert_to_local_time([file_end_utc], local_timezone)[0] if file_end_utc is not None else pd.NaT,
                "DatasetType": dataset_type,
                "CadenceSeconds": cadence_seconds,
                "Selected": include,
            }
        )
    if not selected:
        raise ValueError("No CFACT NetCDF files overlap the requested local-time range.")
    return selected, pd.DataFrame(rows)


__all__ = ["select_files_by_local_timerange"]
