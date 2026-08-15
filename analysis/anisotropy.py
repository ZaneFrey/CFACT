"""Reynolds-stress anisotropy in barycentric coordinates."""

from __future__ import annotations

from typing import Any

from analysis._math import compute_barycentric_coordinates


def barycentric_coordinates(
    u: Any,
    meta_u: dict[str, Any] | None,
    v: Any,
    meta_v: dict[str, Any] | None,
    w: Any,
    meta_w: dict[str, Any] | None,
    time_axis: Any,
    averaging_period_seconds: float = 1.0,
    centered_gliding: bool = True,
):
    """Return time, x_B, and y_B with physically invalid points marked NaN."""

    return compute_barycentric_coordinates(
        u,
        meta_u,
        v,
        meta_v,
        w,
        meta_w,
        time_axis,
        averaging_period_seconds,
        centered_gliding,
    )


__all__ = ["barycentric_coordinates"]
