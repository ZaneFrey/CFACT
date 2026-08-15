from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd

from tools.common import (
    as_native_value,
    convert_to_local_time,
    detect_site_code,
    extract_variable_prefix,
    get_site_definitions,
    make_valid_name,
    normalize_filter_list,
    normalize_nc_inputs,
    normalize_time_bound,
    resolve_site_field,
)

try:
    from netCDF4 import Dataset
except Exception as exc:  # pragma: no cover
    Dataset = None
    NETCDF_IMPORT_ERROR = exc
else:
    NETCDF_IMPORT_ERROR = None


def _require_netcdf4() -> None:
    if Dataset is None:
        raise ImportError(
            "netCDF4 is required for the CFACT Python port. "
            "Run commands through the `cfact` conda environment."
        ) from NETCDF_IMPORT_ERROR


def _parse_name_value_pairs(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    if len(args) % 2 != 0:
        raise ValueError("Name-Value options must be supplied as pairs.")
    merged = dict(kwargs)
    for idx in range(0, len(args), 2):
        merged[str(args[idx])] = args[idx + 1]
    return merged


def _parse_read_options(*args: Any, **kwargs: Any) -> dict[str, Any]:
    options = {
        "SiteCodes": [],
        "VarPrefixes": [],
        "CollapseSamples": "none",
        "LocalTimeZone": "America/Denver",
        "StartTimeLocal": None,
        "EndTimeLocal": None,
    }
    merged = _parse_name_value_pairs(args, kwargs)
    alias_map = {
        "sitecodes": "SiteCodes",
        "varprefixes": "VarPrefixes",
        "variableprefixes": "VarPrefixes",
        "prefixes": "VarPrefixes",
        "collapsesamples": "CollapseSamples",
        "samplecollapse": "CollapseSamples",
        "localtimezone": "LocalTimeZone",
        "timezone": "LocalTimeZone",
        "starttimelocal": "StartTimeLocal",
        "endtimelocal": "EndTimeLocal",
    }
    for key, value in merged.items():
        lookup = alias_map.get(str(key).lower())
        if lookup is None:
            raise ValueError(f'Unknown option "{key}".')
        options[lookup] = value
    options["SiteCodes"] = normalize_filter_list(options["SiteCodes"], force_lower=True)
    options["VarPrefixes"] = normalize_filter_list(options["VarPrefixes"], force_lower=False)
    options["CollapseSamples"] = str(options["CollapseSamples"]).strip().lower()
    if options["CollapseSamples"] not in {"none", "mean", "median", "first"}:
        raise ValueError("CollapseSamples must be one of: none, mean, median, first.")
    options["LocalTimeZone"] = str(options["LocalTimeZone"]).strip()
    options["StartTimeLocal"] = normalize_time_bound(options["StartTimeLocal"], options["LocalTimeZone"])
    options["EndTimeLocal"] = normalize_time_bound(options["EndTimeLocal"], options["LocalTimeZone"])
    return options


def _dataset_info(dataset: Dataset) -> dict[str, Any]:
    return {
        "Dimensions": [{"Name": name, "Length": len(dim)} for name, dim in dataset.dimensions.items()],
        "Variables": [
            {
                "Name": name,
                "Size": tuple(var.shape),
                "Datatype": str(var.dtype),
                "Dimensions": list(var.dimensions),
                "Attributes": [{"Name": attr, "Value": as_native_value(var.getncattr(attr))} for attr in var.ncattrs()],
            }
            for name, var in dataset.variables.items()
        ],
        "Attributes": [{"Name": attr, "Value": as_native_value(dataset.getncattr(attr))} for attr in dataset.ncattrs()],
    }


def _get_variable_info(dataset: Dataset, name: str) -> dict[str, Any]:
    var = dataset.variables[name]
    return {
        "Name": name,
        "Size": tuple(var.shape),
        "Datatype": str(var.dtype),
        "Dimensions": list(var.dimensions),
        "Attributes": [{"Name": attr, "Value": as_native_value(var.getncattr(attr))} for attr in var.ncattrs()],
    }


def _variable_attributes(var: Any) -> dict[str, Any]:
    return {str(attr): as_native_value(var.getncattr(attr)) for attr in var.ncattrs()}


def _infer_time_info(dataset: Dataset, opts: dict[str, Any]) -> dict[str, Any]:
    time_info = {
        "nTime": 0,
        "sampleCount": 1,
        "hasSampleDimension": "sample" in dataset.dimensions,
        "timeIntervalSeconds": float("nan"),
        "nominalSampleRateHz": float("nan"),
        "isSampleCollapsed": opts["CollapseSamples"] != "none",
        "baseTimeUnix": float("nan"),
    }
    if "time" in dataset.dimensions:
        time_info["nTime"] = len(dataset.dimensions["time"])
    if "sample" in dataset.dimensions:
        time_info["sampleCount"] = len(dataset.dimensions["sample"])
    if "time" in dataset.variables:
        attrs = _variable_attributes(dataset.variables["time"])
        interval = attrs.get("interval(sec)")
        if interval is not None:
            time_info["timeIntervalSeconds"] = float(interval)
    if time_info["hasSampleDimension"] and np.isfinite(time_info["timeIntervalSeconds"]) and time_info["timeIntervalSeconds"] > 0:
        time_info["nominalSampleRateHz"] = float(time_info["sampleCount"]) / float(time_info["timeIntervalSeconds"])
    return time_info


def determine_time_selection(nc_file: str, dataset: Dataset, opts: dict[str, Any]) -> dict[str, Any]:
    selection = {
        "requested": False,
        "timeStart": 0,
        "timeCount": None,
        "timeUTC": pd.DatetimeIndex([]),
        "timeLocal": pd.DatetimeIndex([]),
    }
    if opts["StartTimeLocal"] is None and opts["EndTimeLocal"] is None:
        return selection
    if "time" not in dataset.variables or "base_time" not in dataset.variables:
        return selection
    base_time = float(np.asarray(dataset.variables["base_time"][:]).reshape(-1)[0])
    time_raw = np.asarray(dataset.variables["time"][:], dtype=float).reshape(-1)
    time_utc = pd.to_datetime(base_time + time_raw, unit="s", utc=True)
    time_local = convert_to_local_time(time_utc, opts["LocalTimeZone"])
    keep = np.ones(time_local.size, dtype=bool)
    if opts["StartTimeLocal"] is not None:
        keep &= time_local >= opts["StartTimeLocal"]
    if opts["EndTimeLocal"] is not None:
        keep &= time_local <= opts["EndTimeLocal"]
    selection["requested"] = True
    if not np.any(keep):
        selection["timeCount"] = 0
        return selection
    idx = np.flatnonzero(keep)
    selection["timeStart"] = int(idx[0])
    selection["timeCount"] = int(idx[-1] - idx[0] + 1)
    selection["timeUTC"] = pd.DatetimeIndex(time_utc[idx[0] : idx[-1] + 1])
    selection["timeLocal"] = pd.DatetimeIndex(time_local[idx[0] : idx[-1] + 1])
    return selection


def _read_variable_with_time_selection(dataset: Dataset, var_name: str, selection: dict[str, Any]) -> np.ndarray:
    var = dataset.variables[var_name]
    dims = list(var.dimensions)
    if "time" not in dims or not selection["requested"]:
        data = var[:]
    else:
        time_dim = dims.index("time")
        if selection["timeCount"] == 0:
            shape = list(var.shape)
            shape[time_dim] = 0
            return np.empty(shape, dtype=float)
        slices = [slice(None)] * len(dims)
        slices[time_dim] = slice(selection["timeStart"], selection["timeStart"] + selection["timeCount"])
        data = var[tuple(slices)]
    if np.ma.isMaskedArray(data):
        fill = np.nan if np.issubdtype(data.dtype, np.floating) else 0
        data = data.filled(fill)
    return np.asarray(data)


def _maybe_collapse_samples(var_data: np.ndarray, var_meta: dict[str, Any], collapse_mode: str) -> tuple[np.ndarray, dict[str, Any]]:
    if collapse_mode == "none":
        return var_data, var_meta
    dimensions = list(var_meta.get("dimensions", []))
    if "sample" not in dimensions:
        return var_data, var_meta
    sample_dim = dimensions.index("sample")
    original_size = tuple(var_data.shape)
    if collapse_mode == "mean":
        out = np.nanmean(var_data.astype(float), axis=sample_dim)
    elif collapse_mode == "median":
        out = np.nanmedian(var_data.astype(float), axis=sample_dim)
    else:
        out = np.take(var_data, 0, axis=sample_dim)
    out = np.squeeze(out)
    out_meta = dict(var_meta)
    out_meta["originalSize"] = original_size
    out_meta["originalDimensions"] = dimensions
    out_meta["sampleCollapseMode"] = collapse_mode
    out_meta["sampleCount"] = original_size[sample_dim]
    if out.ndim <= 1:
        out = np.asarray(out).reshape(-1)
        out_meta["dimensions"] = ["time"]
    else:
        out_meta["dimensions"] = [dim for idx, dim in enumerate(dimensions) if idx != sample_dim]
    out_meta["size"] = tuple(out.shape)
    return np.asarray(out), out_meta


def _should_read_variable(var_name: str, all_site_codes: list[str], opts: dict[str, Any]) -> bool:
    if var_name in {"base_time", "time"}:
        return True
    site_code = detect_site_code(var_name, all_site_codes)
    if opts["SiteCodes"]:
        if not site_code or site_code not in opts["SiteCodes"]:
            return False
    if not opts["VarPrefixes"]:
        return bool(site_code) or not opts["SiteCodes"]
    prefix = extract_variable_prefix(var_name, all_site_codes)
    return prefix in opts["VarPrefixes"] or var_name in opts["VarPrefixes"]


def _build_site_container(site_defs: list[dict[str, str]]) -> dict[str, Any]:
    sites: dict[str, Any] = {}
    for site in site_defs:
        sites[site["field"]] = {
            "abbr": site["abbr"],
            "ncSuffix": site["ncSuffix"],
            "displayName": site["displayName"],
            "siteType": site["siteType"],
            "data": {},
            "meta": {},
            "varNames": [],
        }
    return sites


def _build_site_summary(sites: dict[str, Any], site_defs: list[dict[str, str]]) -> pd.DataFrame:
    rows = []
    for site in site_defs:
        field = site["field"]
        rows.append(
            {
                "SiteField": field,
                "Abbreviation": site["abbr"],
                "DisplayName": site["displayName"],
                "NumVariables": len(sites[field]["varNames"]),
            }
        )
    return pd.DataFrame(rows).sort_values("NumVariables", ascending=False, ignore_index=True)


def _read_single_cfact_file(nc_file: str, opts: dict[str, Any]) -> dict[str, Any]:
    _require_netcdf4()
    site_defs = get_site_definitions()
    all_site_codes = [site["ncSuffix"] for site in site_defs]
    with Dataset(nc_file) as dataset:
        info = _dataset_info(dataset)
        time_info = _infer_time_info(dataset, opts)
        selection = determine_time_selection(nc_file, dataset, opts)
        cfact = {
            "file": nc_file,
            "info": info,
            "siteDefs": site_defs,
            "globalAttributes": {make_valid_name(attr): as_native_value(dataset.getncattr(attr)) for attr in dataset.ncattrs()},
            "raw": {},
            "meta": {},
            "misc": {},
            "misc_meta": {},
            "sites": _build_site_container(site_defs),
            "timeInfo": time_info,
            "localTimeZone": opts["LocalTimeZone"],
            "nameMap": [],
        }
        for var_name in dataset.variables:
            if not _should_read_variable(var_name, all_site_codes, opts):
                continue
            var = dataset.variables[var_name]
            field_name = make_valid_name(var_name)
            var_data = _read_variable_with_time_selection(dataset, var_name, selection)
            var_meta = {
                "originalName": var_name,
                "size": tuple(var_data.shape),
                "datatype": str(var.dtype),
                "dimensions": list(var.dimensions),
                "attributes": _variable_attributes(var),
            }
            var_data, var_meta = _maybe_collapse_samples(var_data, var_meta, opts["CollapseSamples"])
            cfact["raw"][field_name] = var_data
            cfact["meta"][field_name] = var_meta
            site_code = detect_site_code(var_name, all_site_codes)
            if site_code:
                field = resolve_site_field(site_defs, site_code)
                cfact["sites"][field]["data"][field_name] = var_data
                cfact["sites"][field]["meta"][field_name] = var_meta
                cfact["sites"][field]["varNames"].append(var_name)
                cfact["nameMap"].append(
                    {
                        "NetCDF_Name": var_name,
                        "MATLAB_Field": field_name,
                        "SiteSuffix": site_code,
                        "SiteField": field,
                    }
                )
            else:
                cfact["misc"][field_name] = var_data
                cfact["misc_meta"][field_name] = var_meta
                cfact["nameMap"].append(
                    {
                        "NetCDF_Name": var_name,
                        "MATLAB_Field": field_name,
                        "SiteSuffix": "",
                        "SiteField": "misc",
                    }
                )
        if "base_time" in cfact["misc"] and "time" in cfact["misc"]:
            base_time = float(np.asarray(cfact["misc"]["base_time"]).reshape(-1)[0])
            unix_time = base_time + np.asarray(cfact["misc"]["time"], dtype=float).reshape(-1)
            cfact["time_datetime"] = pd.to_datetime(unix_time, unit="s", utc=True)
            cfact["time_datetime_local"] = convert_to_local_time(cfact["time_datetime"], opts["LocalTimeZone"])
            cfact["timeInfo"]["baseTimeUnix"] = base_time
            cfact["timeInfo"]["nTime"] = int(len(cfact["time_datetime"]))
            cfact["timeInfo"]["nRawSamples"] = int(cfact["timeInfo"]["sampleCount"]) * int(cfact["timeInfo"]["nTime"])
        else:
            cfact["time_datetime"] = pd.DatetimeIndex([])
            cfact["time_datetime_local"] = pd.DatetimeIndex([])
        _add_derived_wind_speed(cfact)
        for site in cfact["sites"].values():
            site["varNames"] = list(dict.fromkeys(site["varNames"]))
        cfact["nameMap"] = pd.DataFrame(cfact["nameMap"])
        cfact["siteSummary"] = _build_site_summary(cfact["sites"], site_defs)
        return cfact


def _is_time_resolved_variable(var_data: Any, var_meta: dict[str, Any], n_time: int) -> bool:
    if "time" in (var_meta.get("dimensions") or []):
        return True
    arr = np.asarray(var_data)
    return arr.ndim == 1 and arr.size == n_time and n_time > 0


def _concat_time_data(a: Any, b: Any, meta: dict[str, Any]) -> np.ndarray:
    a_arr = np.asarray(a)
    b_arr = np.asarray(b)
    if a_arr.ndim <= 1 and b_arr.ndim <= 1:
        return np.concatenate([a_arr.reshape(-1), b_arr.reshape(-1)])
    dimensions = list(meta.get("dimensions", []))
    if "time" not in dimensions:
        raise ValueError("Could not determine time dimension for concatenation.")
    time_dim = dimensions.index("time")
    return np.concatenate([a_arr, b_arr], axis=time_dim)


def _apply_time_index(values: Any, meta: dict[str, Any], idx_keep: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim <= 1:
        return arr.reshape(-1)[idx_keep]
    dimensions = list(meta.get("dimensions", []))
    if "time" not in dimensions:
        raise ValueError("Could not determine time dimension for indexing.")
    time_dim = dimensions.index("time")
    return np.take(arr, idx_keep, axis=time_dim)


def _merge_time_container(a_data: dict[str, Any], a_meta: dict[str, Any], b_data: dict[str, Any], b_meta: dict[str, Any], n_time_b: int) -> tuple[dict[str, Any], dict[str, Any]]:
    for field, b_var in b_data.items():
        if field not in b_meta:
            raise ValueError(f'Missing metadata for variable "{field}".')
        meta = b_meta[field]
        if _is_time_resolved_variable(b_var, meta, n_time_b):
            if field not in a_data:
                raise ValueError(f'Time-resolved variable "{field}" appears in a later file but not in earlier merged data.')
            a_data[field] = _concat_time_data(a_data[field], b_var, meta)
        elif field not in a_data:
            a_data[field] = b_var
            a_meta[field] = meta
    return a_data, a_meta


def _merge_cfact_structs(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    n_time_b = len(b["time_datetime"])
    a["time_datetime"] = pd.DatetimeIndex(list(a["time_datetime"]) + list(b["time_datetime"]))
    a["raw"], a["meta"] = _merge_time_container(a["raw"], a["meta"], b["raw"], b["meta"], n_time_b)
    a["misc"], a["misc_meta"] = _merge_time_container(a["misc"], a["misc_meta"], b["misc"], b["misc_meta"], n_time_b)
    for field in a["sites"]:
        a["sites"][field]["data"], a["sites"][field]["meta"] = _merge_time_container(
            a["sites"][field]["data"],
            a["sites"][field]["meta"],
            b["sites"][field]["data"],
            b["sites"][field]["meta"],
            n_time_b,
        )
        merged_var_names = a["sites"][field]["varNames"] + b["sites"][field]["varNames"]
        a["sites"][field]["varNames"] = list(dict.fromkeys(merged_var_names))
    return a


def _reorder_time_container(data: dict[str, Any], meta: dict[str, Any], idx_keep: np.ndarray, old_n_time: int) -> tuple[dict[str, Any], dict[str, Any]]:
    for field, value in list(data.items()):
        field_meta = meta.get(field, {})
        if _is_time_resolved_variable(value, field_meta, old_n_time):
            data[field] = _apply_time_index(value, field_meta, idx_keep)
    return data, meta


def _collapse_to_time_series(values: Any, meta: dict[str, Any]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.asarray([], dtype=float)
    if arr.ndim <= 1:
        return arr.reshape(-1)
    dimensions = list(meta.get("dimensions", []))
    time_dim = dimensions.index("time") if "time" in dimensions else 0
    axes = [axis for axis in range(arr.ndim) if axis != time_dim]
    for axis in sorted(axes, reverse=True):
        if arr.shape[axis] == 0:
            return np.asarray([], dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            arr = np.nanmean(arr, axis=axis)
    return np.asarray(arr).reshape(-1)


def _add_derived_wind_speed(cfact: dict[str, Any]) -> None:
    for site in cfact["sites"].values():
        fields = list(site["data"].keys())
        for u_field in [name for name in fields if name.startswith("u_")]:
            suffix = u_field[2:]
            v_field = f"v_{suffix}"
            spd_field = f"spd_{suffix}"
            if v_field not in site["data"] or spd_field in site["data"]:
                continue
            u_bar = _collapse_to_time_series(site["data"][u_field], site["meta"][u_field])
            v_bar = _collapse_to_time_series(site["data"][v_field], site["meta"][v_field])
            spd = np.hypot(u_bar, v_bar).reshape(-1)
            meta = {
                "originalName": spd_field,
                "size": tuple(spd.shape),
                "datatype": "double",
                "dimensions": ["time"],
                "attributes": {
                    "long_name": "Derived horizontal wind speed from u and v",
                    "units": "m/s",
                    "derived": "true",
                    "source_u": u_field,
                    "source_v": v_field,
                },
            }
            site["data"][spd_field] = spd
            site["meta"][spd_field] = meta
            site["varNames"].append(spd_field)
            cfact["raw"][spd_field] = spd
            cfact["meta"][spd_field] = meta
    if isinstance(cfact.get("nameMap"), pd.DataFrame):
        rows = []
        for site in cfact["sites"].values():
            for field_name, meta in site["meta"].items():
                if field_name.startswith("spd_") and not (cfact["nameMap"]["MATLAB_Field"] == field_name).any():
                    rows.append(
                        {
                            "NetCDF_Name": field_name,
                            "MATLAB_Field": field_name,
                            "SiteSuffix": site["ncSuffix"],
                            "SiteField": resolve_site_field(cfact["siteDefs"], site["ncSuffix"]),
                        }
                    )
        if rows:
            cfact["nameMap"] = pd.concat([cfact["nameMap"], pd.DataFrame(rows)], ignore_index=True)


def _read_fluxes(nc_input: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
    opts = _parse_read_options(*args, **kwargs)
    script_dir = Path(__file__).resolve().parent.parent
    files = normalize_nc_inputs(nc_input, script_dir=script_dir)
    cfact = _read_single_cfact_file(files[0], opts)
    cfact["file"] = files[0]
    cfact["files"] = files
    cfact["nFiles"] = len(files)
    cfact["localTimeZone"] = opts["LocalTimeZone"]
    cfact["readOptions"] = opts
    cfact["infoPerFile"] = [cfact["info"]]
    file_rows = []
    if len(cfact["time_datetime"]) > 0:
        file_rows.append(
            {
                "File": files[0],
                "StartTimeUTC": cfact["time_datetime"][0],
                "EndTimeUTC": cfact["time_datetime"][-1],
                "StartTimeLocal": cfact["time_datetime_local"][0],
                "EndTimeLocal": cfact["time_datetime_local"][-1],
                "NumSamples": len(cfact["time_datetime"]),
                "SamplesPerTimeStep": int(cfact["timeInfo"]["sampleCount"]),
                "NumRawSamples": int(cfact["timeInfo"].get("nRawSamples", len(cfact["time_datetime"]))),
            }
        )
    else:
        file_rows.append(
            {
                "File": files[0],
                "StartTimeUTC": pd.NaT,
                "EndTimeUTC": pd.NaT,
                "StartTimeLocal": pd.NaT,
                "EndTimeLocal": pd.NaT,
                "NumSamples": 0,
                "SamplesPerTimeStep": int(cfact["timeInfo"]["sampleCount"]),
                "NumRawSamples": 0,
            }
        )
    for file_name in files[1:]:
        this_struct = _read_single_cfact_file(file_name, opts)
        cfact["infoPerFile"].append(this_struct["info"])
        file_rows.append(
            {
                "File": file_name,
                "StartTimeUTC": this_struct["time_datetime"][0] if len(this_struct["time_datetime"]) else pd.NaT,
                "EndTimeUTC": this_struct["time_datetime"][-1] if len(this_struct["time_datetime"]) else pd.NaT,
                "StartTimeLocal": this_struct["time_datetime_local"][0] if len(this_struct["time_datetime_local"]) else pd.NaT,
                "EndTimeLocal": this_struct["time_datetime_local"][-1] if len(this_struct["time_datetime_local"]) else pd.NaT,
                "NumSamples": len(this_struct["time_datetime"]),
                "SamplesPerTimeStep": int(this_struct["timeInfo"]["sampleCount"]),
                "NumRawSamples": int(this_struct["timeInfo"].get("nRawSamples", len(this_struct["time_datetime"]))),
            }
        )
        cfact = _merge_cfact_structs(cfact, this_struct)
    cfact["fileSummary"] = pd.DataFrame(file_rows)
    if len(cfact["time_datetime"]) > 0:
        sorted_idx = np.argsort(cfact["time_datetime"].asi8)
        sorted_times = cfact["time_datetime"][sorted_idx]
        _, unique_positions = np.unique(sorted_times.asi8, return_index=True)
        idx_keep = sorted_idx[np.sort(unique_positions)]
        old_n_time = len(cfact["time_datetime"])
        cfact["time_datetime"] = cfact["time_datetime"][idx_keep]
        cfact["time_datetime_local"] = convert_to_local_time(cfact["time_datetime"], opts["LocalTimeZone"])
        cfact["raw"], cfact["meta"] = _reorder_time_container(cfact["raw"], cfact["meta"], idx_keep, old_n_time)
        cfact["misc"], cfact["misc_meta"] = _reorder_time_container(cfact["misc"], cfact["misc_meta"], idx_keep, old_n_time)
        for field in cfact["sites"]:
            cfact["sites"][field]["data"], cfact["sites"][field]["meta"] = _reorder_time_container(
                cfact["sites"][field]["data"],
                cfact["sites"][field]["meta"],
                idx_keep,
                old_n_time,
            )
        cfact["timeInfo"]["nTime"] = len(cfact["time_datetime"])
    cfact["siteSummary"] = _build_site_summary(cfact["sites"], cfact["siteDefs"])
    return cfact


def _annotate_meta_struct(meta_struct: dict[str, Any], source_dataset: str, utc_field: str, local_field: str, cadence_seconds: int) -> dict[str, Any]:
    for meta in meta_struct.values():
        meta["sourceDataset"] = source_dataset
        meta["timeAxisUTCField"] = utc_field
        meta["timeAxisLocalField"] = local_field
        meta["nativeCadenceSeconds"] = cadence_seconds
    return meta_struct


def _annotate_dataset_metadata(cfact: dict[str, Any], source_dataset: str, utc_field: str, local_field: str, cadence_seconds: int) -> dict[str, Any]:
    cfact["meta"] = _annotate_meta_struct(cfact["meta"], source_dataset, utc_field, local_field, cadence_seconds)
    cfact["misc_meta"] = _annotate_meta_struct(cfact["misc_meta"], source_dataset, utc_field, local_field, cadence_seconds)
    for site in cfact["sites"].values():
        site["meta"] = _annotate_meta_struct(site["meta"], source_dataset, utc_field, local_field, cadence_seconds)
    return cfact


def _read_combined_fluxes(nc_input_20hz: Any, nc_input_5min: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
    merged = _parse_name_value_pairs(args, kwargs)
    five_minute_prefixes = merged.pop("FiveMinuteVarPrefixes", merged.pop("fiveminutevarprefixes", None))
    opts20 = dict(merged)
    opts5 = {key: value for key, value in merged.items() if str(key).lower() not in {"varprefixes", "variableprefixes", "prefixes"}}
    if five_minute_prefixes is not None:
        opts5["VarPrefixes"] = five_minute_prefixes
    has20 = nc_input_20hz not in (None, "", [])
    has5 = nc_input_5min not in (None, "", [])
    if not has20 and not has5:
        raise ValueError("Provide at least one 20 Hz or 5-minute NetCDF input.")
    c20 = _read_fluxes(nc_input_20hz, **opts20) if has20 else None
    c5 = _read_fluxes(nc_input_5min, **opts5) if has5 else None
    if c20 is not None:
        c20 = _annotate_dataset_metadata(c20, "20hz", "time_datetime", "time_datetime_local", 1)
    if c5 is not None:
        c5 = _annotate_dataset_metadata(c5, "5min", "time_datetime_5min", "time_datetime_local_5min", 300)
    cfact = c20 if c20 is not None else c5
    assert cfact is not None
    cfact["time_datetime_5min"] = pd.DatetimeIndex([])
    cfact["time_datetime_local_5min"] = pd.DatetimeIndex([])
    cfact["files20Hz"] = c20["files"] if c20 else []
    cfact["files5min"] = c5["files"] if c5 else []
    cfact["fileSummary20Hz"] = c20["fileSummary"].copy() if c20 else pd.DataFrame()
    cfact["fileSummary5min"] = c5["fileSummary"].copy() if c5 else pd.DataFrame()
    cfact["timeInfo5min"] = c5["timeInfo"] if c5 else {}
    cfact["datasetSources"] = {"has20Hz": has20, "has5min": has5}
    if c5 is not None:
        cfact["time_datetime_5min"] = c5["time_datetime"]
        cfact["time_datetime_local_5min"] = c5["time_datetime_local"]
    if c20 is None and c5 is not None:
        cfact["time_datetime"] = c5["time_datetime"]
        cfact["time_datetime_local"] = c5["time_datetime_local"]
        cfact["timeInfo"] = c5["timeInfo"]
    if c20 is not None and c5 is not None:
        for field, value in c5["raw"].items():
            if field in cfact["raw"]:
                continue
            cfact["raw"][field] = value
            cfact["meta"][field] = c5["meta"][field]
        for field, value in c5["misc"].items():
            if field in cfact["misc"]:
                continue
            cfact["misc"][field] = value
            cfact["misc_meta"][field] = c5["misc_meta"][field]
        for site_field, site in c5["sites"].items():
            target = cfact["sites"][site_field]
            for field, value in site["data"].items():
                if field in target["data"]:
                    continue
                target["data"][field] = value
                target["meta"][field] = site["meta"][field]
                target["varNames"].append(field)
        frames = [df for df in [cfact["fileSummary20Hz"], cfact["fileSummary5min"]] if not df.empty]
        if frames:
            cfact["fileSummary"] = pd.concat(frames, ignore_index=True).sort_values(["StartTimeUTC", "File"], ignore_index=True)
        cfact["files"] = (c20["files"] if c20 else []) + (c5["files"] if c5 else [])
        cfact["nFiles"] = len(cfact["files"])
        cfact["infoPerFile"] = (c20["infoPerFile"] if c20 else []) + (c5["infoPerFile"] if c5 else [])
    elif c20 is None and c5 is not None:
        cfact["fileSummary"] = cfact["fileSummary5min"]
        cfact["files"] = c5["files"]
        cfact["nFiles"] = c5["nFiles"]
        cfact["infoPerFile"] = c5["infoPerFile"]
    else:
        cfact["fileSummary"] = cfact["fileSummary20Hz"]
    cfact["siteSummary"] = _build_site_summary(cfact["sites"], cfact["siteDefs"])
    return cfact


def _parse_combined_read_options(*args: Any, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    merged = _parse_name_value_pairs(args, kwargs)
    five_minute_prefixes = merged.pop("FiveMinuteVarPrefixes", merged.pop("fiveminutevarprefixes", None))
    opts20 = dict(merged)
    opts5 = {key: value for key, value in merged.items() if str(key).lower() not in {"varprefixes", "variableprefixes", "prefixes"}}
    if five_minute_prefixes is not None:
        opts5["VarPrefixes"] = five_minute_prefixes
    return _parse_read_options(**opts20), _parse_read_options(**opts5)


def read_fluxes(
    nc_input: Any,
    *,
    site_codes: Any = None,
    var_prefixes: Any = None,
    collapse_samples: str = "none",
    local_timezone: str = "America/Denver",
    start_time_local: Any = None,
    end_time_local: Any = None,
) -> dict[str, Any]:
    """Read one or more high-rate or five-minute CFACT NetCDF files."""

    return _read_fluxes(
        nc_input,
        SiteCodes=[] if site_codes is None else site_codes,
        VarPrefixes=[] if var_prefixes is None else var_prefixes,
        CollapseSamples=collapse_samples,
        LocalTimeZone=local_timezone,
        StartTimeLocal=start_time_local,
        EndTimeLocal=end_time_local,
    )


def read_combined_fluxes(
    nc_input_20hz: Any,
    nc_input_5min: Any,
    *,
    site_codes: Any = None,
    var_prefixes: Any = None,
    five_minute_var_prefixes: Any = None,
    collapse_samples: str = "none",
    local_timezone: str = "America/Denver",
    start_time_local: Any = None,
    end_time_local: Any = None,
) -> dict[str, Any]:
    """Read and combine high-rate and native five-minute products."""

    return _read_combined_fluxes(
        nc_input_20hz,
        nc_input_5min,
        SiteCodes=[] if site_codes is None else site_codes,
        VarPrefixes=[] if var_prefixes is None else var_prefixes,
        FiveMinuteVarPrefixes=[] if five_minute_var_prefixes is None else five_minute_var_prefixes,
        CollapseSamples=collapse_samples,
        LocalTimeZone=local_timezone,
        StartTimeLocal=start_time_local,
        EndTimeLocal=end_time_local,
    )


__all__ = ["read_combined_fluxes", "read_fluxes"]
