"""Placeholder for proper orthogonal decomposition (POD)."""

from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from pathlib import Path

from analysis.driver_common import driver_parser
from analysis.models import PlotArtifact

SAVE_FIGURES = False


def run(config_path: str | Path | None = None, flag_overrides: dict[str, bool] | None = None) -> list[PlotArtifact]:
    raise NotImplementedError(
        "POD is not implemented. Add a validated snapshot/field decomposition in analysis/pod.py before enabling this driver."
    )


def main() -> None:
    args = driver_parser(__doc__ or "POD placeholder").parse_args()
    try:
        run(args.config)
    except NotImplementedError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
