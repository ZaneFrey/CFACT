from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import pywt

from tools.common import get_meta_dim, get_meta_name, sanitize_series_array


def estimate_time_step_seconds(time_values: Any) -> float:
    idx = pd.DatetimeIndex(time_values)
    if len(idx) < 2:
        return 1.0
    # Pandas may retain datetime64[us] or datetime64[ms] input resolution;
    # normalize explicitly before converting integer ticks to seconds.
    ticks_ns = idx.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    delta = np.diff(ticks_ns.astype(np.float64)) / 1e9
    dt = float(np.nanmedian(delta))
    if not np.isfinite(dt) or dt <= 0:
        return 1.0
    return dt


def estimate_time_rate_hz(time_values: Any) -> float:
    return 1.0 / estimate_time_step_seconds(time_values)


def fill_missing_samples(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    idx = np.arange(arr.size, dtype=float)
    valid = np.isfinite(arr)
    if not np.any(valid):
        return np.zeros_like(arr)
    if np.all(valid):
        return arr
    arr[~valid] = np.interp(idx[~valid], idx[valid], arr[valid])
    return arr


def moving_mean(values: Any, window_samples: int) -> np.ndarray:
    return pd.Series(np.asarray(values, dtype=float).reshape(-1)).rolling(window_samples, center=True, min_periods=1).mean().to_numpy()


def moving_var(values: Any, window_samples: int) -> np.ndarray:
    return pd.Series(np.asarray(values, dtype=float).reshape(-1)).rolling(window_samples, center=True, min_periods=1).var(ddof=1).to_numpy()


def expand_raw_series(values: Any, meta: dict[str, Any] | None, t_second: Any, require_sample_dim: bool = True) -> tuple[np.ndarray, np.ndarray, float, pd.DatetimeIndex]:
    arr = np.asarray(values, dtype=float)
    time_axis = pd.DatetimeIndex(t_second)
    n_time = len(time_axis)
    if arr.ndim <= 1:
        series = arr.reshape(-1)
        if series.size != n_time:
            raise ValueError(
                f'Vector variable has {series.size} samples but time axis has {n_time} for "{get_meta_name(meta)}".'
            )
        rep_idx = np.arange(n_time)
        return series, rep_idx, estimate_time_rate_hz(time_axis), time_axis
    time_dim = get_meta_dim(meta or {}, "time")
    sample_dim = get_meta_dim(meta or {}, "sample")
    if time_dim is None:
        matches = [idx for idx, size in enumerate(arr.shape) if size == n_time]
        time_dim = matches[-1] if matches else None
    if time_dim is None:
        raise ValueError(f'Could not determine the time dimension for "{get_meta_name(meta)}".')
    if sample_dim is None and require_sample_dim:
        raise ValueError(f'Could not determine the sample dimension for "{get_meta_name(meta)}".')
    if sample_dim is None:
        axes = [idx for idx in range(arr.ndim) if idx != time_dim]
        collapsed = arr.astype(float)
        for axis in sorted(axes, reverse=True):
            collapsed = np.nanmean(collapsed, axis=axis)
        series = np.asarray(collapsed).reshape(-1)
        return series, np.arange(n_time), estimate_time_rate_hz(time_axis), time_axis
    axes_other = [idx for idx in range(arr.ndim) if idx not in (sample_dim, time_dim)]
    collapsed = arr.astype(float)
    for axis in sorted(axes_other, reverse=True):
        collapsed = np.nanmean(collapsed, axis=axis)
    moved = np.moveaxis(collapsed, (sample_dim, time_dim), (0, 1))
    moved = np.squeeze(moved)
    if moved.ndim != 2:
        raise ValueError(f'Expected a 2D [sample,time] array after collapsing other dimensions for "{get_meta_name(meta)}".')
    if moved.shape[1] != n_time:
        if moved.shape[0] == n_time:
            moved = moved.T
        else:
            raise ValueError(f'Time dimension does not match the requested time axis for "{get_meta_name(meta)}".')
    n_sample = moved.shape[0]
    raw = moved.reshape(-1, order="F")
    rep_idx = np.arange(n_time) * n_sample + int(math.ceil(n_sample / 2.0)) - 1
    seconds_per_step = estimate_time_step_seconds(time_axis)
    sample_rate_hz = n_sample / seconds_per_step
    sample_offsets = (-0.5 * seconds_per_step) + (((np.arange(n_sample) + 0.5) / n_sample) * seconds_per_step)
    base_seconds = time_axis.view("int64") / 1e9
    t_matrix = base_seconds[None, :] + sample_offsets[:, None]
    t_raw = pd.to_datetime(t_matrix.reshape(-1, order="F"), unit="s", utc=True).tz_convert(time_axis.tz)
    return raw, rep_idx.astype(int), sample_rate_hz, pd.DatetimeIndex(t_raw)


def expand_native_sample_stream_wavelet(values: Any, meta: dict[str, Any] | None, t_second: Any) -> tuple[np.ndarray, float, pd.DatetimeIndex]:
    arr = np.asarray(values, dtype=float)
    time_axis = pd.DatetimeIndex(t_second)
    n_time = len(time_axis)
    dt_second = estimate_time_step_seconds(time_axis)
    if arr.ndim <= 1:
        series = arr.reshape(-1)
        if series.size != n_time:
            raise ValueError(
                f'Vector variable has {series.size} samples but expected {n_time} for "{get_meta_name(meta)}".'
            )
        return series, 1.0 / dt_second, time_axis
    time_dim = get_meta_dim(meta or {}, "time")
    sample_dim = get_meta_dim(meta or {}, "sample")
    if time_dim is None:
        matches = [idx for idx, size in enumerate(arr.shape) if size == n_time]
        time_dim = matches[-1] if matches else None
    if time_dim is None:
        raise ValueError(f'Could not determine the time dimension for the wavelet calculation for "{get_meta_name(meta)}".')
    if sample_dim is None:
        axes = [idx for idx in range(arr.ndim) if idx != time_dim]
        collapsed = arr.astype(float)
        for axis in sorted(axes, reverse=True):
            collapsed = np.nanmean(collapsed, axis=axis)
        series = np.asarray(collapsed).reshape(-1)
        return series, 1.0 / dt_second, time_axis
    axes_other = [idx for idx in range(arr.ndim) if idx not in (sample_dim, time_dim)]
    collapsed = arr.astype(float)
    for axis in sorted(axes_other, reverse=True):
        collapsed = np.nanmean(collapsed, axis=axis)
    moved = np.moveaxis(collapsed, (sample_dim, time_dim), (0, 1))
    moved = np.squeeze(moved)
    if moved.ndim != 2:
        raise ValueError(f'Expected a 2D [sample,time] array after collapsing other dimensions for "{get_meta_name(meta)}".')
    if moved.shape[1] != n_time:
        if moved.shape[0] == n_time:
            moved = moved.T
        else:
            raise ValueError(f'Time dimension does not match the requested time axis for "{get_meta_name(meta)}".')
    n_sample = moved.shape[0]
    raw = moved.reshape(-1, order="F")
    sample_rate_hz = n_sample / dt_second
    second_offsets = pd.to_timedelta(np.arange(n_sample, dtype=float) / sample_rate_hz, unit="s")
    repeated_time = time_axis.repeat(n_sample)
    tiled_offsets = np.tile(second_offsets.to_numpy(), n_time)
    t_native = pd.DatetimeIndex(repeated_time + pd.to_timedelta(tiled_offsets))
    return raw, sample_rate_hz, t_native


def align_raw_series(x_raw: Any, rep_idx_x: np.ndarray, sample_rate_x: float, t_raw_x: Any, y_raw: Any, rep_idx_y: np.ndarray, sample_rate_y: float, t_raw_y: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, pd.DatetimeIndex]:
    x_raw = np.asarray(x_raw, dtype=float).reshape(-1)
    y_raw = np.asarray(y_raw, dtype=float).reshape(-1)
    t_raw_x = pd.DatetimeIndex(t_raw_x)
    t_raw_y = pd.DatetimeIndex(t_raw_y)
    same = (
        x_raw.size == y_raw.size
        and rep_idx_x.size == rep_idx_y.size
        and np.array_equal(rep_idx_x, rep_idx_y)
        and x_raw.size == t_raw_x.size
        and y_raw.size == t_raw_y.size
        and np.array_equal(t_raw_x.asi8, t_raw_y.asi8)
        and abs(sample_rate_x - sample_rate_y) <= max(1e-9, 1e-9 * max(sample_rate_x, sample_rate_y))
    )
    if same:
        return x_raw, y_raw, rep_idx_x, sample_rate_x, t_raw_x
    if sample_rate_x <= sample_rate_y:
        return x_raw, resample_to_time_base(y_raw, t_raw_y, t_raw_x), rep_idx_x, sample_rate_x, t_raw_x
    return resample_to_time_base(x_raw, t_raw_x, t_raw_y), y_raw, rep_idx_y, sample_rate_y, t_raw_y


def align_three_raw_series(u_raw: Any, rep_idx_u: np.ndarray, sample_rate_u: float, t_raw_u: Any, v_raw: Any, rep_idx_v: np.ndarray, sample_rate_v: float, t_raw_v: Any, w_raw: Any, rep_idx_w: np.ndarray, sample_rate_w: float, t_raw_w: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, pd.DatetimeIndex]:
    sample_rates = [sample_rate_u, sample_rate_v, sample_rate_w]
    idx_target = int(np.argmin(sample_rates))
    raws = [np.asarray(u_raw, dtype=float).reshape(-1), np.asarray(v_raw, dtype=float).reshape(-1), np.asarray(w_raw, dtype=float).reshape(-1)]
    rep_idx = [rep_idx_u, rep_idx_v, rep_idx_w][idx_target]
    time_target = pd.DatetimeIndex([t_raw_u, t_raw_v, t_raw_w][idx_target])
    for idx, t_raw in enumerate([t_raw_u, t_raw_v, t_raw_w]):
        if idx == idx_target:
            continue
        raws[idx] = resample_to_time_base(raws[idx], pd.DatetimeIndex(t_raw), time_target)
    return raws[0], raws[1], raws[2], rep_idx, float(sample_rates[idx_target]), time_target


def resample_to_time_base(values: Any, t_source: Any, t_target: Any) -> np.ndarray:
    source = np.asarray(values, dtype=float).reshape(-1)
    source_time = pd.DatetimeIndex(t_source).asi8.astype(np.float64) / 1e9
    target_time = pd.DatetimeIndex(t_target).asi8.astype(np.float64) / 1e9
    valid = np.isfinite(source)
    if not np.any(valid):
        return np.full(target_time.shape, np.nan)
    source_time = source_time[valid]
    source = source[valid]
    source_time, unique_idx = np.unique(source_time, return_index=True)
    source = source[unique_idx]
    if source_time.size == 1:
        return np.full(target_time.shape, float(source[0]))
    return np.interp(target_time, source_time, source)


def compute_window_stat(values: Any, meta: dict[str, Any] | None, t_second: Any, avg_period_seconds: float = 1.0, glide: bool = True, stat_name: str = "mean") -> tuple[pd.DatetimeIndex, np.ndarray]:
    x_raw, rep_idx, sample_rate_hz, t_raw = expand_raw_series(values, meta, t_second)
    if x_raw.size == 0:
        return pd.DatetimeIndex([]), np.asarray([])
    window_samples = max(1, int(round(avg_period_seconds * sample_rate_hz)))
    stat_name = str(stat_name).lower()
    if stat_name == "variance":
        stat_name = "var"
    if glide:
        if stat_name == "mean":
            y_raw = moving_mean(x_raw, window_samples)
        elif stat_name == "var":
            y_raw = moving_var(x_raw, window_samples)
        else:
            raise ValueError('stat_name must be "mean" or "var".')
        return pd.DatetimeIndex(t_second), y_raw[rep_idx]
    return apply_block_stat(x_raw, t_raw, window_samples, stat_name)


def apply_block_stat(values: Any, t_raw: Any, block_samples: int, stat_name: str) -> tuple[pd.DatetimeIndex, np.ndarray]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    time_axis = pd.DatetimeIndex(t_raw)
    n_block = int(math.ceil(arr.size / block_samples))
    out = np.full(n_block, np.nan)
    t_out: list[pd.Timestamp] = []
    for idx in range(n_block):
        idx1 = idx * block_samples
        idx2 = min((idx + 1) * block_samples, arr.size)
        block = arr[idx1:idx2]
        if stat_name == "mean":
            out[idx] = np.nanmean(block)
        else:
            out[idx] = np.nanvar(block, ddof=1)
        t_out.append(time_axis[idx1] + (time_axis[idx2 - 1] - time_axis[idx1]) / 2)
    return pd.DatetimeIndex(t_out), out


def compute_window_covariance(x: Any, meta_x: dict[str, Any] | None, y: Any, meta_y: dict[str, Any] | None, t_second: Any, avg_period_seconds: float = 1.0, glide: bool = True) -> tuple[pd.DatetimeIndex, np.ndarray]:
    x_raw, rep_idx_x, sample_rate_x, t_raw_x = expand_raw_series(x, meta_x, t_second)
    y_raw, rep_idx_y, sample_rate_y, t_raw_y = expand_raw_series(y, meta_y, t_second)
    x_raw, y_raw, rep_idx, sample_rate_hz, t_raw = align_raw_series(x_raw, rep_idx_x, sample_rate_x, t_raw_x, y_raw, rep_idx_y, sample_rate_y, t_raw_y)
    window_samples = max(1, int(round(avg_period_seconds * sample_rate_hz)))
    if glide:
        mean_x = moving_mean(x_raw, window_samples)
        mean_y = moving_mean(y_raw, window_samples)
        mean_xy = moving_mean(x_raw * y_raw, window_samples)
        cov_raw = mean_xy - mean_x * mean_y
        return pd.DatetimeIndex(t_second), cov_raw[rep_idx]
    return apply_block_covariance(x_raw, y_raw, t_raw, window_samples)


def apply_block_covariance(x_raw: Any, y_raw: Any, t_raw: Any, block_samples: int) -> tuple[pd.DatetimeIndex, np.ndarray]:
    x_arr = np.asarray(x_raw, dtype=float).reshape(-1)
    y_arr = np.asarray(y_raw, dtype=float).reshape(-1)
    time_axis = pd.DatetimeIndex(t_raw)
    n_block = int(math.ceil(x_arr.size / block_samples))
    out = np.full(n_block, np.nan)
    t_out: list[pd.Timestamp] = []
    for idx in range(n_block):
        idx1 = idx * block_samples
        idx2 = min((idx + 1) * block_samples, x_arr.size)
        xb = x_arr[idx1:idx2]
        yb = y_arr[idx1:idx2]
        out[idx] = np.nanmean(xb * yb) - np.nanmean(xb) * np.nanmean(yb)
        t_out.append(time_axis[idx1] + (time_axis[idx2 - 1] - time_axis[idx1]) / 2)
    return pd.DatetimeIndex(t_out), out


def collapse_to_second_series(values: Any, meta: dict[str, Any] | None, t_second: Any) -> tuple[np.ndarray, float]:
    arr = np.asarray(values, dtype=float)
    n_time = len(pd.DatetimeIndex(t_second))
    dt_second = estimate_time_step_seconds(t_second)
    if arr.ndim <= 1:
        out = arr.reshape(-1)
        if out.size != n_time:
            raise ValueError(f"Collapsed second-level series has {out.size} samples but expected {n_time}.")
        return out, dt_second
    time_dim = get_meta_dim(meta or {}, "time")
    if time_dim is None:
        matches = [idx for idx, size in enumerate(arr.shape) if size == n_time]
        time_dim = matches[-1] if matches else None
    if time_dim is None:
        raise ValueError("Could not determine the time dimension for autocorrelation.")
    axes = [idx for idx in range(arr.ndim) if idx != time_dim]
    collapsed = arr
    for axis in sorted(axes, reverse=True):
        collapsed = np.nanmean(collapsed, axis=axis)
    out = np.asarray(collapsed).reshape(-1)
    if out.size != n_time:
        raise ValueError(f"Collapsed second-level series has {out.size} samples but expected {n_time}.")
    return out, dt_second


def lagged_dot(a: Any, b: Any, max_lag_samples: int) -> np.ndarray:
    a_arr = np.asarray(a, dtype=float).reshape(-1)
    b_arr = np.asarray(b, dtype=float).reshape(-1)
    if a_arr.size != b_arr.size:
        raise ValueError("Input vectors for lagged_dot must have the same length.")
    n = a_arr.size
    nfft = 1 << int(math.ceil(math.log2(max(1, 2 * n - 1))))
    conv = np.fft.ifft(np.fft.fft(a_arr[::-1], nfft) * np.fft.fft(b_arr, nfft)).real[: (2 * n - 1)]
    idx = n - 1 + np.arange(max_lag_samples + 1)
    return conv[idx]


def compute_autocorrelation(values: Any, meta: dict[str, Any] | None, t_second: Any, max_lag_seconds: float = 600.0) -> tuple[np.ndarray, np.ndarray]:
    x_series, dt = collapse_to_second_series(values, meta, t_second)
    max_lag_samples = min(int(math.floor(max_lag_seconds / dt)), max(0, x_series.size - 1))
    lag_seconds = np.arange(max_lag_samples + 1, dtype=float) * dt
    x = np.asarray(x_series, dtype=float).reshape(-1)
    mask = np.isfinite(x)
    x_zero = x.copy()
    x_zero[~mask] = 0.0
    x2 = x_zero**2
    mask_float = mask.astype(float)
    count = lagged_dot(mask_float, mask_float, max_lag_samples)
    sum1 = lagged_dot(x_zero, mask_float, max_lag_samples)
    sum2 = lagged_dot(mask_float, x_zero, max_lag_samples)
    sumsq1 = lagged_dot(x2, mask_float, max_lag_samples)
    sumsq2 = lagged_dot(mask_float, x2, max_lag_samples)
    cross = lagged_dot(x_zero, x_zero, max_lag_samples)
    out = np.full(max_lag_samples + 1, np.nan)
    valid = count >= 2
    mu1 = np.full_like(count, np.nan)
    mu2 = np.full_like(count, np.nan)
    mu1[valid] = sum1[valid] / count[valid]
    mu2[valid] = sum2[valid] / count[valid]
    numer = cross - count * mu1 * mu2
    var1 = sumsq1 - count * (mu1**2)
    var2 = sumsq2 - count * (mu2**2)
    denom = np.sqrt(var1 * var2)
    good = valid & np.isfinite(denom) & (denom > 0)
    out[good] = numer[good] / denom[good]
    if out.size:
        out[0] = 1.0
    return lag_seconds, out


def _compute_lagged_correlation(x: np.ndarray, y: np.ndarray, max_lag_samples: int) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float).reshape(-1)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    if x_arr.size != y_arr.size:
        raise ValueError("Input vectors for lagged correlation must have the same length.")
    if x_arr.size == 0:
        return np.asarray([], dtype=float)
    max_lag_samples = int(min(max(0, max_lag_samples), x_arr.size - 1))
    mask_x = np.isfinite(x_arr)
    mask_y = np.isfinite(y_arr)
    x_zero = x_arr.copy()
    y_zero = y_arr.copy()
    x_zero[~mask_x] = 0.0
    y_zero[~mask_y] = 0.0
    x2 = x_zero**2
    y2 = y_zero**2
    mask_x_float = mask_x.astype(float)
    mask_y_float = mask_y.astype(float)
    count = lagged_dot(mask_x_float, mask_y_float, max_lag_samples)
    sum_x = lagged_dot(x_zero, mask_y_float, max_lag_samples)
    sum_y = lagged_dot(mask_x_float, y_zero, max_lag_samples)
    sumsq_x = lagged_dot(x2, mask_y_float, max_lag_samples)
    sumsq_y = lagged_dot(mask_x_float, y2, max_lag_samples)
    cross = lagged_dot(x_zero, y_zero, max_lag_samples)
    out = np.full(max_lag_samples + 1, np.nan)
    valid = count >= 2
    mu_x = np.full_like(count, np.nan)
    mu_y = np.full_like(count, np.nan)
    mu_x[valid] = sum_x[valid] / count[valid]
    mu_y[valid] = sum_y[valid] / count[valid]
    numer = cross - count * mu_x * mu_y
    var_x = sumsq_x - count * (mu_x**2)
    var_y = sumsq_y - count * (mu_y**2)
    denom = np.sqrt(var_x * var_y)
    good = valid & np.isfinite(denom) & (denom > 0)
    out[good] = numer[good] / denom[good]
    return out


def _integrate_to_zero_crossing(lag_seconds: np.ndarray, correlation: np.ndarray) -> float:
    lag = np.asarray(lag_seconds, dtype=float).reshape(-1)
    rho = np.asarray(correlation, dtype=float).reshape(-1)
    if lag.size != rho.size or lag.size < 2:
        return float("nan")
    if not np.isfinite(rho[0]):
        return float("nan")
    sign0 = np.sign(rho[0])
    if sign0 == 0:
        return 0.0
    last_idx = 0
    for idx in range(1, rho.size):
        if not np.isfinite(rho[idx]):
            break
        if rho[idx] == 0:
            return float(np.trapezoid(rho[: idx + 1], lag[: idx + 1]))
        if np.sign(rho[idx]) != sign0:
            lag_left = lag[idx - 1]
            lag_right = lag[idx]
            rho_left = rho[idx - 1]
            rho_right = rho[idx]
            if not np.isfinite(rho_left) or not np.isfinite(rho_right) or lag_right <= lag_left or rho_right == rho_left:
                return float("nan")
            tau_cross = lag_left - rho_left * (lag_right - lag_left) / (rho_right - rho_left)
            base_integral = float(np.trapezoid(rho[:idx], lag[:idx])) if idx > 1 else 0.0
            tail_integral = 0.5 * rho_left * (tau_cross - lag_left)
            return base_integral + tail_integral
        last_idx = idx
    return float("nan") if last_idx < rho.size - 1 else float("nan")


def _rolling_sum_for_starts(values: np.ndarray, window_length: int, start_indices: np.ndarray) -> np.ndarray:
    prefix = np.concatenate(([0.0], np.cumsum(np.asarray(values, dtype=float))))
    return prefix[start_indices + window_length] - prefix[start_indices]


def _correlation_vector_at_lag(
    x_pad: np.ndarray,
    y_pad: np.ndarray,
    start_indices: np.ndarray,
    window_samples: int,
    lag_samples: int,
) -> np.ndarray:
    if lag_samples < 0:
        raise ValueError("lag_samples must be non-negative.")
    pair_count = window_samples - lag_samples
    if pair_count < 2:
        return np.full(start_indices.size, np.nan)
    if lag_samples == 0:
        x_lag = x_pad
        y_lag = y_pad
    else:
        x_lag = x_pad[:-lag_samples]
        y_lag = y_pad[lag_samples:]
    valid = np.isfinite(x_lag) & np.isfinite(y_lag)
    valid_float = valid.astype(float)
    x_zero = np.where(valid, x_lag, 0.0)
    y_zero = np.where(valid, y_lag, 0.0)
    count = _rolling_sum_for_starts(valid_float, pair_count, start_indices)
    sum_x = _rolling_sum_for_starts(x_zero, pair_count, start_indices)
    sum_y = _rolling_sum_for_starts(y_zero, pair_count, start_indices)
    sum_x2 = _rolling_sum_for_starts(x_zero * x_zero, pair_count, start_indices)
    sum_y2 = _rolling_sum_for_starts(y_zero * y_zero, pair_count, start_indices)
    sum_xy = _rolling_sum_for_starts(x_zero * y_zero, pair_count, start_indices)
    out = np.full(start_indices.size, np.nan)
    good = count >= 2
    if not np.any(good):
        return out
    mu_x = np.full(start_indices.size, np.nan)
    mu_y = np.full(start_indices.size, np.nan)
    mu_x[good] = sum_x[good] / count[good]
    mu_y[good] = sum_y[good] / count[good]
    numer = sum_xy - count * mu_x * mu_y
    var_x = sum_x2 - count * (mu_x**2)
    var_y = sum_y2 - count * (mu_y**2)
    denom = np.sqrt(var_x * var_y)
    valid_corr = good & np.isfinite(denom) & (denom > 0)
    out[valid_corr] = numer[valid_corr] / denom[valid_corr]
    return out


def compute_integral_timescale(
    x: Any,
    meta_x: dict[str, Any] | None,
    y: Any,
    meta_y: dict[str, Any] | None,
    t_second: Any,
    avg_period_seconds: float = 300.0,
    glide: bool = True,
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    if not glide:
        raise ValueError("Integral timescale analysis requires glide=True.")
    x_raw, rep_idx_x, sample_rate_x, t_raw_x = expand_raw_series(x, meta_x, t_second, require_sample_dim=False)
    y_raw, rep_idx_y, sample_rate_y, t_raw_y = expand_raw_series(y, meta_y, t_second, require_sample_dim=False)
    x_raw, y_raw, rep_idx, sample_rate_hz, _ = align_raw_series(
        x_raw, rep_idx_x, sample_rate_x, t_raw_x, y_raw, rep_idx_y, sample_rate_y, t_raw_y
    )
    x_raw = sanitize_series_array(x_raw, meta_x)
    y_raw = sanitize_series_array(y_raw, meta_y)
    x_raw = np.asarray(x_raw, dtype=float).reshape(-1)
    y_raw = np.asarray(y_raw, dtype=float).reshape(-1)
    rep_idx = np.asarray(rep_idx, dtype=int).reshape(-1)
    if x_raw.size != y_raw.size:
        raise ValueError("Integral timescale raw inputs must align to the same number of samples.")
    if x_raw.size == 0 or rep_idx.size == 0:
        return pd.DatetimeIndex([]), np.asarray([], dtype=float)
    dt = 1.0 / float(sample_rate_hz)
    window_samples = max(3, int(round(float(avg_period_seconds) * sample_rate_hz)))
    if window_samples % 2 == 0:
        window_samples += 1
    max_lag_samples = min(window_samples - 1, x_raw.size - 1)
    if max_lag_samples < 1:
        return pd.DatetimeIndex(t_second), np.full(rep_idx.size, np.nan)
    half_window = window_samples // 2
    out = np.full(rep_idx.size, np.nan)
    x_pad = np.pad(x_raw, (half_window, half_window), constant_values=np.nan)
    y_pad = np.pad(y_raw, (half_window, half_window), constant_values=np.nan)
    start_indices = np.clip(rep_idx, 0, x_raw.size - 1)
    rho_prev = _correlation_vector_at_lag(x_pad, y_pad, start_indices, window_samples, 0)
    sign0 = np.sign(rho_prev)
    done = ~np.isfinite(rho_prev)
    out[sign0 == 0] = 0.0
    done |= sign0 == 0
    integral = np.zeros(rep_idx.size, dtype=float)
    for lag_idx in range(1, max_lag_samples + 1):
        active = ~done
        if not np.any(active):
            break
        rho_curr = _correlation_vector_at_lag(x_pad, y_pad, start_indices, window_samples, lag_idx)
        tau_prev = (lag_idx - 1) * dt
        tau_curr = lag_idx * dt
        invalid = active & ~np.isfinite(rho_curr)
        done[invalid] = True
        exact_zero = active & np.isfinite(rho_curr) & (rho_curr == 0)
        out[exact_zero] = integral[exact_zero] + 0.5 * (rho_prev[exact_zero] + rho_curr[exact_zero]) * (tau_curr - tau_prev)
        done[exact_zero] = True
        crossing = active & np.isfinite(rho_curr) & (np.sign(rho_curr) != sign0)
        crossing &= rho_curr != 0
        stable_crossing = crossing & (rho_curr != rho_prev)
        tau_cross = np.full(rep_idx.size, np.nan)
        tau_cross[stable_crossing] = tau_prev - rho_prev[stable_crossing] * (tau_curr - tau_prev) / (
            rho_curr[stable_crossing] - rho_prev[stable_crossing]
        )
        out[stable_crossing] = integral[stable_crossing] + 0.5 * rho_prev[stable_crossing] * (tau_cross[stable_crossing] - tau_prev)
        done[crossing] = True
        same_sign = active & ~done & np.isfinite(rho_curr) & (np.sign(rho_curr) == sign0)
        integral[same_sign] += 0.5 * (rho_prev[same_sign] + rho_curr[same_sign]) * (tau_curr - tau_prev)
        rho_prev[same_sign] = rho_curr[same_sign]
    return pd.DatetimeIndex(t_second), out


def _apply_log_binning(frequency_hz: Any, spectrum: Any, bins_per_decade: float, signed: bool = False) -> tuple[np.ndarray, np.ndarray]:
    f = np.asarray(frequency_hz, dtype=float).reshape(-1)
    s = np.asarray(spectrum, dtype=float).reshape(-1)
    valid = np.isfinite(f) & np.isfinite(s) & (f > 0)
    if not signed:
        valid &= s > 0
    f = f[valid]
    s = s[valid]
    if f.size < 3:
        return f, s
    log_f = np.log10(f)
    log_span = float(np.nanmax(log_f) - np.nanmin(log_f))
    if not np.isfinite(log_span) or log_span <= 0:
        return f, s
    n_bins = max(1, int(math.ceil(log_span * bins_per_decade)))
    edges = np.linspace(np.nanmin(log_f), np.nanmax(log_f), n_bins + 1)
    out_f: list[float] = []
    out_s: list[float] = []
    for idx in range(n_bins):
        if idx < n_bins - 1:
            in_bin = (log_f >= edges[idx]) & (log_f < edges[idx + 1])
        else:
            in_bin = (log_f >= edges[idx]) & (log_f <= edges[idx + 1])
        if not np.any(in_bin):
            continue
        out_f.append(float(10 ** np.nanmean(log_f[in_bin])))
        out_s.append(float(np.nanmean(s[in_bin])))
    return np.asarray(out_f), np.asarray(out_s)


def compute_1d_spectrum(values: Any, meta: dict[str, Any] | None, t_second: Any, ApplyLogBinning: bool = False, LogBinsPerDecade: float = 12) -> tuple[np.ndarray, np.ndarray]:
    raw, _, sample_rate_hz, _ = expand_raw_series(values, meta, t_second, require_sample_dim=False)
    raw = fill_missing_samples(raw)
    if raw.size % 2 == 0:
        raw = raw[:-1]
    n = raw.size
    if n < 3:
        raise ValueError("The input series is too short to compute a 1D spectrum.")
    n_f = (n - 1) // 2
    fft_vals = np.fft.fft(raw) / n
    abs_f2 = fft_vals * np.conj(fft_vals)
    frequency_hz = sample_rate_hz * np.arange(1, n_f + 1, dtype=float) / n
    delta_f = sample_rate_hz / n
    spectral_density = 2 * np.real(abs_f2[1 : n_f + 1]) / delta_f
    if ApplyLogBinning:
        return _apply_log_binning(frequency_hz, spectral_density, LogBinsPerDecade, signed=False)
    return frequency_hz, np.asarray(spectral_density, dtype=float)


def compute_cospectrum(x: Any, meta_x: dict[str, Any] | None, y: Any, meta_y: dict[str, Any] | None, t_second: Any, ApplyLogBinning: bool = False, LogBinsPerDecade: float = 12) -> tuple[np.ndarray, np.ndarray, float]:
    x_raw, rep_idx_x, sample_rate_x, t_raw_x = expand_raw_series(x, meta_x, t_second)
    y_raw, rep_idx_y, sample_rate_y, t_raw_y = expand_raw_series(y, meta_y, t_second)
    x_raw, y_raw, _, sample_rate_hz, _ = align_raw_series(x_raw, rep_idx_x, sample_rate_x, t_raw_x, y_raw, rep_idx_y, sample_rate_y, t_raw_y)
    x_raw = sanitize_series_array(x_raw, meta_x)
    y_raw = sanitize_series_array(y_raw, meta_y)
    if np.count_nonzero(np.isfinite(x_raw)) < 2 or np.count_nonzero(np.isfinite(y_raw)) < 2:
        return np.asarray([]), np.asarray([]), float("nan")
    x_raw = fill_missing_samples(x_raw)
    y_raw = fill_missing_samples(y_raw)
    x_raw = x_raw - np.nanmean(x_raw)
    y_raw = y_raw - np.nanmean(y_raw)
    if x_raw.size % 2 == 0:
        x_raw = x_raw[:-1]
        y_raw = y_raw[:-1]
    n = x_raw.size
    if n < 3:
        return np.asarray([]), np.asarray([]), float("nan")
    n_f = (n - 1) // 2
    covariance_xy = float(np.nanmean(x_raw * y_raw))
    fx = np.fft.fft(x_raw) / n
    fy = np.fft.fft(y_raw) / n
    cross = fx * np.conj(fy)
    frequency_hz = sample_rate_hz * np.arange(1, n_f + 1, dtype=float) / n
    delta_f = sample_rate_hz / n
    cospectral_density = 2 * np.real(cross[1 : n_f + 1]) / delta_f
    if ApplyLogBinning:
        frequency_hz, cospectral_density = _apply_log_binning(frequency_hz, cospectral_density, LogBinsPerDecade, signed=True)
    return frequency_hz, np.asarray(cospectral_density, dtype=float), covariance_xy


def compute_wavelet_analysis(values: Any, meta: dict[str, Any] | None, t_second: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
    raw, sample_rate_hz, t_native = expand_native_sample_stream_wavelet(values, meta, t_second)
    raw = fill_missing_samples(raw)
    raw = raw - np.nanmean(raw)
    n = raw.size
    if n < 8:
        raise ValueError("Wavelet analysis requires at least 8 samples.")
    # PyWavelets does not provide MATLAB's exact `amor` wavelet, so use a
    # denser analytic-Morlet approximation and a MATLAB-like log-spaced bank.
    wavelet = pywt.ContinuousWavelet("cmor1.5-1.0")
    max_freq = sample_rate_hz / 2.0
    min_freq = max(sample_rate_hz / n, 1.0 / max(n, 1))
    voices_per_octave = 10
    n_octaves = max(1.0, float(np.log2(max_freq / min_freq)))
    n_frequency = int(np.ceil(voices_per_octave * n_octaves)) + 1
    n_frequency = min(128, max(32, n_frequency))
    frequencies = np.geomspace(max_freq, min_freq, num=n_frequency)
    scales = pywt.frequency2scale(wavelet, frequencies / sample_rate_hz)
    coeffs, frequency_hz = pywt.cwt(raw, scales, wavelet, sampling_period=1.0 / sample_rate_hz, method="fft")
    wt_magnitude = np.abs(coeffs)
    wavelet_power = np.nanmean(np.abs(coeffs) ** 2, axis=1)
    wavelet_energy = 1e-3 * wavelet_power / np.maximum(np.asarray(frequency_hz, dtype=float), np.finfo(float).eps)
    return np.asarray(frequency_hz, dtype=float), np.asarray(wavelet_energy, dtype=float), wt_magnitude, t_native


def compute_specific_humidity_flux(rhoVapor_gm3: Any, temperatureDegC: Any, pressureMb: Any, wRhoVaporFlux_gm3_ms: Any) -> tuple[np.ndarray, np.ndarray]:
    rho_vapor = np.asarray(rhoVapor_gm3, dtype=float).reshape(-1) / 1000.0
    temperature_k = np.asarray(temperatureDegC, dtype=float).reshape(-1) + 273.15
    pressure_pa = np.asarray(pressureMb, dtype=float).reshape(-1) * 100.0
    w_rho_vapor_flux = np.asarray(wRhoVaporFlux_gm3_ms, dtype=float).reshape(-1) / 1000.0
    if not (rho_vapor.size == temperature_k.size == pressure_pa.size == w_rho_vapor_flux.size):
        raise ValueError("All moisture-flux inputs must have the same length.")
    rv = 461.5
    epsilon = 0.622
    vapor_pressure = rho_vapor * rv * temperature_k
    denom = pressure_pa - (1 - epsilon) * vapor_pressure
    q = np.full(rho_vapor.size, np.nan)
    wq = np.full(rho_vapor.size, np.nan)
    valid = np.isfinite(rho_vapor) & np.isfinite(temperature_k) & np.isfinite(pressure_pa) & np.isfinite(w_rho_vapor_flux) & (temperature_k > 0) & (pressure_pa > 0) & (denom > 0)
    q[valid] = epsilon * vapor_pressure[valid] / denom[valid]
    dqdrho = np.full(rho_vapor.size, np.nan)
    dqdrho[valid] = epsilon * pressure_pa[valid] * rv * temperature_k[valid] / (denom[valid] ** 2)
    wq[valid] = dqdrho[valid] * w_rho_vapor_flux[valid]
    return q, wq


def compute_vertical_gradient_lagrange(z: Any, F: Any) -> np.ndarray:
    z_arr = np.asarray(z, dtype=float).reshape(-1)
    f_arr = np.asarray(F, dtype=float)
    if z_arr.size < 3:
        raise ValueError("At least 3 heights are required to compute a vertical gradient.")
    if f_arr.shape[0] != z_arr.size:
        raise ValueError("The first dimension of F must match the number of heights in z.")
    if np.unique(z_arr).size != z_arr.size:
        raise ValueError("The height vector z must contain distinct values.")
    out = np.full_like(f_arr, np.nan, dtype=float)
    for idx in range(z_arr.size):
        if idx == 0:
            stencil = slice(0, 3)
        elif idx == z_arr.size - 1:
            stencil = slice(z_arr.size - 3, z_arr.size)
        else:
            stencil = slice(idx - 1, idx + 2)
        nodes = z_arr[stencil]
        z1, z2, z3 = nodes
        z_eval = z_arr[idx]
        weights = np.array(
            [
                (2 * z_eval - z2 - z3) / ((z1 - z2) * (z1 - z3)),
                (2 * z_eval - z1 - z3) / ((z2 - z1) * (z2 - z3)),
                (2 * z_eval - z1 - z2) / ((z3 - z1) * (z3 - z2)),
            ]
        )
        out[idx, ...] = np.tensordot(weights, f_arr[stencil, ...], axes=(0, 0))
    return out


def _reynolds_stress_to_barycentric(uu: float, vv: float, ww: float, uv: float, uw: float, vw: float) -> tuple[float, float]:
    comps = np.asarray([uu, vv, ww, uv, uw, vw], dtype=float)
    if not np.all(np.isfinite(comps)):
        return float("nan"), float("nan")
    r = np.array([[uu, uv, uw], [uv, vv, vw], [uw, vw, ww]], dtype=float)
    r = 0.5 * (r + r.T)
    k = 0.5 * np.trace(r)
    if not np.isfinite(k) or k <= np.finfo(float).eps * max(1.0, abs(np.trace(r))):
        return float("nan"), float("nan")
    b = r / (2 * k) - np.eye(3) / 3.0
    b = 0.5 * (b + b.T)
    eigvals = np.sort(np.real(np.linalg.eigvals(b)))[::-1]
    if eigvals.size != 3 or not np.all(np.isfinite(eigvals)):
        return float("nan"), float("nan")
    c1 = eigvals[0] - eigvals[1]
    c3 = 3 * eigvals[2] + 1
    xb = float(np.clip(c1 + 0.5 * c3, 0.0, 1.0))
    yb = float(np.clip((math.sqrt(3.0) / 2.0) * c3, 0.0, math.sqrt(3.0) / 2.0))
    return xb, yb


def compute_barycentric_coordinates(u: Any, meta_u: dict[str, Any] | None, v: Any, meta_v: dict[str, Any] | None, w: Any, meta_w: dict[str, Any] | None, t_second: Any, avg_period_seconds: float = 1.0, glide: bool = True) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    u_raw, rep_idx_u, sample_rate_u, t_raw_u = expand_raw_series(u, meta_u, t_second)
    v_raw, rep_idx_v, sample_rate_v, t_raw_v = expand_raw_series(v, meta_v, t_second)
    w_raw, rep_idx_w, sample_rate_w, t_raw_w = expand_raw_series(w, meta_w, t_second)
    u_raw, v_raw, w_raw, rep_idx, sample_rate_hz, t_raw = align_three_raw_series(
        u_raw, rep_idx_u, sample_rate_u, t_raw_u, v_raw, rep_idx_v, sample_rate_v, t_raw_v, w_raw, rep_idx_w, sample_rate_w, t_raw_w
    )
    u_raw = fill_missing_samples(sanitize_series_array(u_raw, meta_u))
    v_raw = fill_missing_samples(sanitize_series_array(v_raw, meta_v))
    w_raw = fill_missing_samples(sanitize_series_array(w_raw, meta_w))
    window_samples = max(1, int(round(avg_period_seconds * sample_rate_hz)))
    if glide:
        mean_u = moving_mean(u_raw, window_samples)
        mean_v = moving_mean(v_raw, window_samples)
        mean_w = moving_mean(w_raw, window_samples)
        uu = moving_mean(u_raw * u_raw, window_samples) - mean_u * mean_u
        vv = moving_mean(v_raw * v_raw, window_samples) - mean_v * mean_v
        ww = moving_mean(w_raw * w_raw, window_samples) - mean_w * mean_w
        uv = moving_mean(u_raw * v_raw, window_samples) - mean_u * mean_v
        uw = moving_mean(u_raw * w_raw, window_samples) - mean_u * mean_w
        vw = moving_mean(v_raw * w_raw, window_samples) - mean_v * mean_w
        xb_out = np.full(rep_idx.size, np.nan)
        yb_out = np.full(rep_idx.size, np.nan)
        for out_idx, raw_idx in enumerate(rep_idx):
            xb_out[out_idx], yb_out[out_idx] = _reynolds_stress_to_barycentric(
                uu[raw_idx], vv[raw_idx], ww[raw_idx], uv[raw_idx], uw[raw_idx], vw[raw_idx]
            )
        return pd.DatetimeIndex(t_second), xb_out, yb_out
    return apply_block_barycentric(u_raw, v_raw, w_raw, t_raw, window_samples)


def apply_block_barycentric(u_raw: Any, v_raw: Any, w_raw: Any, t_raw: Any, block_samples: int) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    u_arr = np.asarray(u_raw, dtype=float).reshape(-1)
    v_arr = np.asarray(v_raw, dtype=float).reshape(-1)
    w_arr = np.asarray(w_raw, dtype=float).reshape(-1)
    time_axis = pd.DatetimeIndex(t_raw)
    n_block = int(math.ceil(u_arr.size / block_samples))
    xb = np.full(n_block, np.nan)
    yb = np.full(n_block, np.nan)
    t_out: list[pd.Timestamp] = []
    for idx in range(n_block):
        idx1 = idx * block_samples
        idx2 = min((idx + 1) * block_samples, u_arr.size)
        ub = u_arr[idx1:idx2]
        vb = v_arr[idx1:idx2]
        wb = w_arr[idx1:idx2]
        mean_u = np.nanmean(ub)
        mean_v = np.nanmean(vb)
        mean_w = np.nanmean(wb)
        xb[idx], yb[idx] = _reynolds_stress_to_barycentric(
            np.nanmean(ub * ub) - mean_u * mean_u,
            np.nanmean(vb * vb) - mean_v * mean_v,
            np.nanmean(wb * wb) - mean_w * mean_w,
            np.nanmean(ub * vb) - mean_u * mean_v,
            np.nanmean(ub * wb) - mean_u * mean_w,
            np.nanmean(vb * wb) - mean_v * mean_w,
        )
        t_out.append(time_axis[idx1] + (time_axis[idx2 - 1] - time_axis[idx1]) / 2)
    return pd.DatetimeIndex(t_out), xb, yb


def compute_tke_transport_flux(u: Any, meta_u: dict[str, Any] | None, v: Any, meta_v: dict[str, Any] | None, w: Any, meta_w: dict[str, Any] | None, t_second: Any, avg_period_seconds: float = 1.0, glide: bool = True) -> tuple[pd.DatetimeIndex, np.ndarray]:
    u_raw, rep_idx_u, sample_rate_u, t_raw_u = expand_raw_series(u, meta_u, t_second)
    v_raw, rep_idx_v, sample_rate_v, t_raw_v = expand_raw_series(v, meta_v, t_second)
    w_raw, rep_idx_w, sample_rate_w, t_raw_w = expand_raw_series(w, meta_w, t_second)
    u_raw, v_raw, w_raw, rep_idx, sample_rate_hz, t_raw = align_three_raw_series(
        u_raw, rep_idx_u, sample_rate_u, t_raw_u, v_raw, rep_idx_v, sample_rate_v, t_raw_v, w_raw, rep_idx_w, sample_rate_w, t_raw_w
    )
    window_samples = max(1, int(round(avg_period_seconds * sample_rate_hz)))
    if glide:
        mean_u = moving_mean(u_raw, window_samples)
        mean_v = moving_mean(v_raw, window_samples)
        mean_w = moving_mean(w_raw, window_samples)
        e_raw = 0.5 * ((u_raw - mean_u) ** 2 + (v_raw - mean_v) ** 2 + (w_raw - mean_w) ** 2)
        mean_e = moving_mean(e_raw, window_samples)
        mean_we = moving_mean(w_raw * e_raw, window_samples)
        w_e = mean_we - mean_w * mean_e
        return pd.DatetimeIndex(t_second), w_e[rep_idx]
    return apply_block_tke_transport(u_raw, v_raw, w_raw, t_raw, window_samples)


def apply_block_tke_transport(u_raw: Any, v_raw: Any, w_raw: Any, t_raw: Any, block_samples: int) -> tuple[pd.DatetimeIndex, np.ndarray]:
    u_arr = np.asarray(u_raw, dtype=float).reshape(-1)
    v_arr = np.asarray(v_raw, dtype=float).reshape(-1)
    w_arr = np.asarray(w_raw, dtype=float).reshape(-1)
    time_axis = pd.DatetimeIndex(t_raw)
    n_block = int(math.ceil(u_arr.size / block_samples))
    out = np.full(n_block, np.nan)
    t_out: list[pd.Timestamp] = []
    for idx in range(n_block):
        idx1 = idx * block_samples
        idx2 = min((idx + 1) * block_samples, u_arr.size)
        ub = u_arr[idx1:idx2]
        vb = v_arr[idx1:idx2]
        wb = w_arr[idx1:idx2]
        mean_u = np.nanmean(ub)
        mean_v = np.nanmean(vb)
        mean_w = np.nanmean(wb)
        e = 0.5 * ((ub - mean_u) ** 2 + (vb - mean_v) ** 2 + (wb - mean_w) ** 2)
        out[idx] = np.nanmean(wb * e) - mean_w * np.nanmean(e)
        t_out.append(time_axis[idx1] + (time_axis[idx2 - 1] - time_axis[idx1]) / 2)
    return pd.DatetimeIndex(t_out), out
