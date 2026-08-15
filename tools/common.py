from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SITE_DEFINITIONS: list[dict[str, str]] = [
    {
        "ncSuffix": "dcs",
        "abbr": "DCSS",
        "field": "DeerCreekSupersite",
        "displayName": "Deer Creek Supersite",
        "siteType": "supersite",
    },
    {
        "ncSuffix": "prs",
        "abbr": "PRSS",
        "field": "ProvoRiverSupersite",
        "displayName": "Provo River Supersite",
        "siteType": "supersite",
    },
    {"ncSuffix": "up", "abbr": "UP", "field": "UpperProvo", "displayName": "Upper Provo", "siteType": "satellite"},
    {"ncSuffix": "cc", "abbr": "CC", "field": "CenterCreek", "displayName": "Center Creek", "siteType": "satellite"},
    {"ncSuffix": "lc", "abbr": "LC", "field": "LakeCreek", "displayName": "Lake Creek", "siteType": "satellite"},
    {"ncSuffix": "dc", "abbr": "DC", "field": "DanielsCanyon", "displayName": "Daniels Canyon", "siteType": "satellite"},
    {"ncSuffix": "mw", "abbr": "MW", "field": "MidwayLane", "displayName": "Midway Lane", "siteType": "satellite"},
    {"ncSuffix": "sp", "abbr": "SP", "field": "SouthPivot", "displayName": "South Pivot", "siteType": "satellite"},
    {"ncSuffix": "sh", "abbr": "SH", "field": "SoldierHollow", "displayName": "Soldier Hollow", "siteType": "satellite"},
    {"ncSuffix": "pc", "abbr": "PC", "field": "PineCanyon", "displayName": "Pine Canyon", "siteType": "satellite"},
    {"ncSuffix": "mh", "abbr": "MH", "field": "MemorialHill", "displayName": "Memorial Hill", "siteType": "satellite"},
    {
        "ncSuffix": "np",
        "abbr": "NP",
        "field": "NorthPivot",
        "displayName": "North Pivot Profiling Site",
        "siteType": "profiling",
    },
]


def make_valid_name(name: str) -> str:
    sanitized = re.sub(r"\W", "_", str(name))
    if not sanitized:
        sanitized = "x"
    if sanitized[0].isdigit():
        sanitized = f"x_{sanitized}"
    return sanitized


def get_site_definitions() -> list[dict[str, str]]:
    return [dict(site) for site in SITE_DEFINITIONS]


def resolve_site_field(site_defs: list[dict[str, str]], site_code: str) -> str:
    site_code = str(site_code).strip().lower()
    for site in site_defs:
        if site["ncSuffix"].lower() == site_code or site["abbr"].lower() == site_code:
            return site["field"]
    raise KeyError(f'Unknown siteCode "{site_code}".')


def normalize_to_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        return [str(value)]
    if isinstance(value, np.ndarray):
        return [str(v) for v in value.reshape(-1).tolist()]
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value]
    return [str(value)]


def normalize_filter_list(value: Any, force_lower: bool = False) -> list[str]:
    items = [str(v).strip() for v in normalize_to_list(value) if str(v).strip()]
    if any(v.lower() == "all" for v in items):
        return []
    if force_lower:
        items = [v.lower() for v in items]
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def resolve_existing_path(in_path: str | Path, script_dir: str | Path | None = None) -> Path:
    path = Path(in_path)
    if path.exists():
        return path.resolve()
    if script_dir is not None:
        candidate = Path(script_dir) / path
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not resolve input path: {in_path}")


def normalize_nc_inputs(nc_input: Any, script_dir: str | Path | None = None) -> list[str]:
    files: list[str] = []
    seen: set[str] = set()
    for entry in normalize_to_list(nc_input):
        full_path = resolve_existing_path(entry, script_dir)
        if full_path.is_dir():
            listing = sorted(full_path.glob("*.nc"), key=lambda p: p.name.lower())
            if not listing:
                raise FileNotFoundError(f"Folder contains no .nc files: {full_path}")
            for item in listing:
                key = str(item.resolve())
                if key not in seen:
                    seen.add(key)
                    files.append(key)
        elif full_path.is_file():
            if full_path.suffix.lower() != ".nc":
                raise ValueError(f"Input file is not a .nc file: {full_path}")
            key = str(full_path.resolve())
            if key not in seen:
                seen.add(key)
                files.append(key)
        else:
            raise FileNotFoundError(f"Input is neither a valid file nor folder: {entry}")
    if not files:
        raise ValueError("No NetCDF files were found.")
    return files


def normalize_time_bound(value: Any, local_timezone: str) -> pd.Timestamp | None:
    if value is None or value == []:
        return None
    if isinstance(value, pd.Timestamp):
        stamp = value
    else:
        stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        return stamp.tz_localize(local_timezone)
    return stamp.tz_convert(local_timezone)


def convert_to_local_time(time_utc: Iterable[Any], local_timezone: str) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(time_utc)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    return idx.tz_convert(local_timezone)


def detect_site_code(var_name: str, site_codes: Iterable[str]) -> str:
    tokens = [token for token in re.split(r"_+", str(var_name).lower()) if token]
    codes = {code.lower() for code in site_codes}
    for token in reversed(tokens):
        if token in codes:
            return token
    return ""


def extract_variable_prefix(var_name: str, site_codes: Iterable[str]) -> str:
    site_code = detect_site_code(var_name, site_codes)
    if not site_code:
        return str(var_name)
    pattern = rf"^(.*?)(?:_(\d+(?:_\d+)?m))?_{re.escape(site_code)}$"
    match = re.match(pattern, str(var_name))
    if not match:
        return str(var_name)
    return match.group(1)


def get_meta_dim(meta: dict[str, Any] | None, dim_name: str) -> int | None:
    if not meta:
        return None
    dimensions = meta.get("dimensions") or []
    for idx, name in enumerate(dimensions):
        if str(name) == dim_name:
            return idx
    return None


def get_meta_name(meta: dict[str, Any] | None) -> str:
    if not meta:
        return "unknown variable"
    return str(meta.get("originalName") or "unknown variable")


def get_time_axis_fields(meta: dict[str, Any] | None) -> tuple[str, str]:
    utc_field = "time_datetime"
    local_field = "time_datetime_local"
    if meta:
        utc_field = str(meta.get("timeAxisUTCField") or utc_field)
        local_field = str(meta.get("timeAxisLocalField") or local_field)
    return utc_field, local_field


def get_variable_time_axis(cfact: dict[str, Any], meta: dict[str, Any] | None = None) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex, str, str]:
    utc_field, local_field = get_time_axis_fields(meta)
    time_utc = cfact.get(utc_field)
    if time_utc is None:
        time_utc = cfact.get("time_datetime")
    if time_utc is None:
        time_utc = pd.DatetimeIndex([])
    time_local = cfact.get(local_field)
    if time_local is None:
        time_local = cfact.get("time_datetime_local")
    if time_local is None:
        time_local = pd.DatetimeIndex([])
    return pd.DatetimeIndex(time_utc), pd.DatetimeIndex(time_local), utc_field, local_field


def height_tag_to_value(height_tag: str) -> float:
    if str(height_tag).lower() == "single":
        return float("nan")
    txt = str(height_tag).lower().replace("m", "").replace("_", ".")
    try:
        return float(txt)
    except ValueError:
        return float("inf")


def format_height_label(height_tag: str) -> str:
    if str(height_tag).lower() == "single":
        return "single level"
    return f"{str(height_tag).replace('m', '').replace('_', '.')} m"


def collect_master_height_tags(stats: Iterable[dict[str, Any]]) -> list[str]:
    tags: list[str] = []
    for stat in stats:
        for series in stat.get("series", []):
            tag = str(series.get("heightTag", "single"))
            if tag not in tags:
                tags.append(tag)
    if not tags:
        return ["single"]
    tags.sort(key=height_tag_to_value)
    return tags


def build_height_colormap(n_height: int, colormap: str = "Blues") -> np.ndarray:
    cmap = plt.get_cmap(colormap)
    if n_height <= 1:
        return np.array([cmap(1.0)[:3]])
    # Use only the darkest 75% of Blues and assign the darkest blue to the
    # lowest height by sampling from dark to lighter shades.
    return np.array([cmap(v)[:3] for v in np.linspace(1.0, 0.5, n_height)])


def sanitize_series_array(values: Any, meta: dict[str, Any] | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    attrs = (meta or {}).get("attributes", {})
    for key in ("_FillValue", "missing_value", "FillValue"):
        if key not in attrs:
            continue
        fill_value = np.asarray(attrs[key], dtype=float)
        if fill_value.size == 0:
            continue
        fill_scalar = float(fill_value.reshape(-1)[0])
        if math.isfinite(fill_scalar):
            arr[np.isclose(arr, fill_scalar, atol=max(1.0, abs(fill_scalar)) * np.finfo(float).eps * 10)] = np.nan
    arr[np.abs(arr) > 1e35] = np.nan
    return arr


def ensure_datetime_index(values: Iterable[Any], timezone: str | None = None) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(values)
    if idx.tz is None and timezone:
        idx = idx.tz_localize(timezone)
    return idx


def build_log_bins(frequency_hz: np.ndarray, bins_per_decade: float) -> np.ndarray:
    log_frequency = np.log10(frequency_hz)
    log_span = float(np.nanmax(log_frequency) - np.nanmin(log_frequency))
    n_bins = max(1, int(math.ceil(log_span * float(bins_per_decade))))
    return np.linspace(np.nanmin(log_frequency), np.nanmax(log_frequency), n_bins + 1)


def as_native_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return value.reshape(-1)[0].item()
        return value.tolist()
    if isinstance(value, (np.generic,)):
        return value.item()
    return value
