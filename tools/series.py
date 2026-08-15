from __future__ import annotations

import re

import numpy as np

from tools.common import get_variable_time_axis, height_tag_to_value, resolve_site_field


def _subset_time_dimension(values, meta, time_mask, n_time_expected):
    arr = np.asarray(values)
    if time_mask is None:
        return arr.reshape(-1) if arr.ndim <= 1 else arr
    mask = np.asarray(time_mask, dtype=bool).reshape(-1)
    if n_time_expected is not None and n_time_expected != mask.size:
        raise ValueError(f"Time mask has {mask.size} samples but the variable time axis has {n_time_expected}.")
    if arr.ndim <= 1:
        if arr.size != mask.size:
            raise ValueError(f"Vector variable has {arr.size} samples but the mask has {mask.size}.")
        return arr.reshape(-1)[mask]
    dimensions = list((meta or {}).get("dimensions") or [])
    time_dim = dimensions.index("time") if "time" in dimensions else next((idx for idx, size in enumerate(arr.shape) if size == mask.size), None)
    if time_dim is None:
        raise ValueError(f'Could not determine the time dimension for variable "{(meta or {}).get("originalName", "unknown")}".')
    return np.take(arr, np.flatnonzero(mask), axis=time_dim)


def _apply_height_selection(series, height_request):
    if height_request is None:
        return series
    if isinstance(height_request, np.ndarray):
        req = height_request
    else:
        req = np.asarray(height_request if isinstance(height_request, (list, tuple)) else [height_request], dtype=float)
    if req.size == 0:
        heights = np.array([item["heightValue"] for item in series], dtype=float)
        idx = np.where(np.isfinite(heights) & np.isclose(heights, 2.0))[0]
        return [series[int(idx[0])]] if idx.size else series
    if req.size == 1 and np.isnan(req[0]):
        return series
    keep = []
    requested = np.unique(req)
    for item in series:
        if np.any(np.isclose(item["heightValue"], requested, atol=1e-9, equal_nan=False)):
            keep.append(item)
    return keep


def collect_height_series(cfact, site_code, prefix, height_request=np.nan, time_mask=None):
    site_field = resolve_site_field(cfact["siteDefs"], site_code)
    site = cfact["sites"][site_field]
    series = []
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+(?:_\d+)?m)_{re.escape(str(site_code).lower())}$")
    fallback_candidates = [f"{prefix}_{str(site_code).lower()}", f"{str(prefix).lower()}_{str(site_code).lower()}"]
    for field_name, values in site["data"].items():
        match = pattern.match(field_name)
        if not match:
            continue
        height_tag = match.group(1)
        meta = site["meta"][field_name]
        _, time_local, _, _ = get_variable_time_axis(cfact, meta)
        data = _subset_time_dimension(values, meta, time_mask, len(time_local))
        series.append(
            {
                "heightTag": height_tag,
                "heightValue": height_tag_to_value(height_tag),
                "varName": field_name,
                "data": data,
                "meta": meta,
            }
        )
    if not series:
        for name in fallback_candidates:
            if name not in site["data"]:
                continue
            meta = site["meta"][name]
            _, time_local, _, _ = get_variable_time_axis(cfact, meta)
            series.append(
                {
                    "heightTag": "single",
                    "heightValue": float("nan"),
                    "varName": name,
                    "data": _subset_time_dimension(site["data"][name], meta, time_mask, len(time_local)),
                    "meta": meta,
                }
            )
            break
    if not series:
        raise ValueError(f'No variables matching prefix "{prefix}" were found for site "{site_code}".')
    series = _apply_height_selection(series, height_request)
    if not series:
        raise ValueError(f'No heights matched the request for prefix "{prefix}" at site "{site_code}".')
    series.sort(key=lambda item: item["heightValue"])
    return series


__all__ = ["collect_height_series"]
