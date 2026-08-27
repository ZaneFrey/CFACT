
"""Statistical moments, friction-velocity, and stability driver."""

from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from pathlib import Path

from analysis.driver_common import (
    align_height_series,
    artifact_from_figure,
    covariance_height_series,
    driver_parser,
    load_data,
    load_driver_config,
    plot_height_series,
    resolve_flags,
    variance_height_series,
)
from analysis.models import PlotArtifact
from analysis.tke import friction_velocity
from tools.plotting import plot_height_histograms
from tools.series import collect_height_series
from tools.style import apply_style

PLOT_U_VARIANCE = False
PLOT_V_VARIANCE = False
PLOT_W_VARIANCE = False
PLOT_PDFS = False
PLOT_FRICTION_VELOCITY = False
PLOT_Z_OVER_L = False
SAVE_FIGURES = False


def run(config_path: str | Path | None = None, flag_overrides: dict[str, bool] | None = None) -> list[PlotArtifact]:
    flags = resolve_flags(
        {
            "plot_u_variance": PLOT_U_VARIANCE,
            "plot_v_variance": PLOT_V_VARIANCE,
            "plot_w_variance": PLOT_W_VARIANCE,
            "plot_pdfs": PLOT_PDFS,
            "plot_friction_velocity": PLOT_FRICTION_VELOCITY,
            "plot_z_over_l": PLOT_Z_OVER_L,
            "save_figures": SAVE_FIGURES,
        },
        flag_overrides,
    )
    if flags["plot_z_over_l"]:
        raise NotImplementedError(
            "Unsupported statistics flag: plot_z_over_l. z/L is a documented placeholder."
        )
    enabled = [name for name, value in flags.items() if name.startswith("plot_") and value]
    if not enabled:
        return []
    config = load_driver_config(config_path)
    prefixes = [component for component in ("u", "v", "w") if flags[f"plot_{component}_variance"]]
    if flags["plot_friction_velocity"]:
        prefixes.extend(["u", "v", "w"])
    if flags["plot_pdfs"]:
        prefixes.extend(["u", "v", "w", "tc"])
    data = load_data(config, list(dict.fromkeys(prefixes)))
    artifacts: list[PlotArtifact] = []
    for component in ("u", "v", "w"):
        if not flags[f"plot_{component}_variance"]:
            continue
        time_axis, series = variance_height_series(data, config, component)
        artifacts.append(
            plot_height_series(
                config,
                f"{component}_variance",
                f"{config.site.upper()}: {component} variance",
                rf"$\sigma^2_{{{component}}}$ [m$^2$ s$^{{-2}}$]",
                time_axis,
                series,
                save_figures=flags["save_figures"],
            )
        )
    if flags["plot_pdfs"]:
        apply_style(config.figure)
        stats = []
        for prefix, label in (("u", "u"), ("v", "v"), ("w", "w"), ("tc", "T")):
            stats.append({"varName": label, "displayName": label, "series": collect_height_series(data, config.site, prefix)})
        figure = plot_height_histograms(stats, f"{config.site.upper()}: probability distributions")
        artifacts.append(artifact_from_figure(config, "pdfs", figure, flags["save_figures"]))
    if flags["plot_friction_velocity"]:
        time_axis, uw = covariance_height_series(data, config, "u", "w")
        _, vw = covariance_height_series(data, config, "v", "w")
        uw, vw = align_height_series(uw, vw)
        series = [
            {**uw_entry, "data": friction_velocity(uw_entry["data"], vw_entry["data"])}
            for uw_entry, vw_entry in zip(uw, vw)
        ]
        artifacts.append(
            plot_height_series(
                config,
                "friction_velocity",
                f"{config.site.upper()}: friction velocity",
                r"$u_*$ [m s$^{-1}$]",
                time_axis,
                series,
                save_figures=flags["save_figures"],
            )
        )
    return artifacts


def main() -> None:
    args = driver_parser(__doc__ or "Statistical analysis").parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
