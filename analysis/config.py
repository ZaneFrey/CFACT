"""Validated YAML configuration for every analysis driver."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd
import yaml

from tools.common import SITE_DEFINITIONS


@dataclass(frozen=True, slots=True)
class FigureConfig:
    output_format: str = "png"
    overwrite: bool = True
    dpi: int = 180
    width: float = 10.0
    panel_height: float = 2.6
    font_family: str = "DejaVu Sans"
    title_size: float = 12.0
    label_size: float = 10.0
    tick_size: float = 9.0
    legend_size: float = 9.0
    line_width: float = 1.3
    grid: bool = True
    height_colormap: str = "Blues"


@dataclass(frozen=True, slots=True)
class AnalysisConfig:
    site: str
    timezone: str
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    averaging_period_seconds: float
    centered_gliding: bool
    data_dir: Path
    output_dir: Path
    figure: FigureConfig = field(default_factory=FigureConfig)

    @property
    def high_rate_pattern(self) -> str:
        return "*hr*.nc"

    @property
    def five_minute_pattern(self) -> str:
        return "*5min*.nc"


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a YAML mapping.")
    return value


def _local_timestamp(value: Any, timezone: str, name: str) -> pd.Timestamp:
    try:
        stamp = pd.Timestamp(value)
    except Exception as exc:
        raise ValueError(f"{name} must be a valid date/time.") from exc
    try:
        return stamp.tz_localize(timezone) if stamp.tzinfo is None else stamp.tz_convert(timezone)
    except Exception as exc:
        raise ValueError(f"{name} is invalid in time zone {timezone!r}.") from exc


def load_config(config_path: str | Path | None = None) -> AnalysisConfig:
    """Load and validate a configuration, resolving paths beside its YAML file."""

    path = Path(config_path) if config_path is not None else Path(__file__).with_name("config.yaml")
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {path}")
    with path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    raw = _require_mapping(raw, "configuration")

    site = str(raw.get("site", "")).strip().lower()
    valid_sites = {entry["ncSuffix"] for entry in SITE_DEFINITIONS}
    if site not in valid_sites:
        raise ValueError(f"site must be one of: {', '.join(sorted(valid_sites))}.")

    timezone = str(raw.get("timezone", "")).strip()
    try:
        ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError(f"Unknown IANA time zone: {timezone!r}.") from exc
    start_time = _local_timestamp(raw.get("start_time"), timezone, "start_time")
    end_time = _local_timestamp(raw.get("end_time"), timezone, "end_time")
    if end_time <= start_time:
        raise ValueError("end_time must be later than start_time.")

    averaging_period_seconds = float(raw.get("averaging_period_seconds", 0))
    if not (averaging_period_seconds > 0):
        raise ValueError("averaging_period_seconds must be greater than zero.")

    data_dir = (path.parent / str(raw.get("data_dir", ""))).resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Configured data directory does not exist: {data_dir}")
    output_dir = (path.parent / str(raw.get("output_dir", "outputs"))).resolve()

    figure_raw = _require_mapping(raw.get("figure", {}), "figure")
    output_format = str(figure_raw.get("format", "png")).lower().lstrip(".")
    if output_format not in {"png", "pdf", "svg"}:
        raise ValueError("figure.format must be one of: png, pdf, svg.")
    dpi = int(figure_raw.get("dpi", 180))
    width = float(figure_raw.get("width", 10.0))
    panel_height = float(figure_raw.get("panel_height", 2.6))
    if dpi <= 0 or width <= 0 or panel_height <= 0:
        raise ValueError("figure dpi, width, and panel_height must be positive.")
    figure = FigureConfig(
        output_format=output_format,
        overwrite=bool(figure_raw.get("overwrite", True)),
        dpi=dpi,
        width=width,
        panel_height=panel_height,
        font_family=str(figure_raw.get("font_family", "DejaVu Sans")),
        title_size=float(figure_raw.get("title_size", 12)),
        label_size=float(figure_raw.get("label_size", 10)),
        tick_size=float(figure_raw.get("tick_size", 9)),
        legend_size=float(figure_raw.get("legend_size", 9)),
        line_width=float(figure_raw.get("line_width", 1.3)),
        grid=bool(figure_raw.get("grid", True)),
        height_colormap=str(figure_raw.get("height_colormap", "Blues")),
    )
    return AnalysisConfig(
        site=site,
        timezone=timezone,
        start_time=start_time,
        end_time=end_time,
        averaging_period_seconds=averaging_period_seconds,
        centered_gliding=bool(raw.get("centered_gliding", True)),
        data_dir=data_dir,
        output_dir=output_dir,
        figure=figure,
    )


__all__ = ["AnalysisConfig", "FigureConfig", "load_config"]
