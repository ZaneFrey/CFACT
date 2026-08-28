from __future__ import annotations

import analysis.driver_correlations as correlations
import analysis.driver_fluxes as fluxes
import analysis.driver_metdata as metdata
import analysis.driver_quadrant as quadrant
import analysis.driver_radiation as radiation
import analysis.driver_stats as stats
import analysis.driver_timescales as timescales
import analysis.driver_tke as tke


def _plot_controls(module):
    return {name for name in vars(module) if name.startswith("PLOT_")}


def test_plot_controls_are_partitioned_by_driver():
    assert _plot_controls(correlations) == {"PLOT_AUTOCORRELATION"}
    assert _plot_controls(quadrant) == {"PLOT_QUADRANT_SCATTER", "PLOT_QUADRANT_JOINT_PDF"}
    assert _plot_controls(timescales) == {"PLOT_INTEGRAL_TIMESCALE"}
    assert _plot_controls(radiation) == {"PLOT_RADIATION"}
    assert _plot_controls(fluxes) == {"PLOT_REYNOLDS_FLUXES", "PLOT_MOISTURE_FLUXES"}
    assert {"PLOT_FRICTION_VELOCITY", "PLOT_Z_OVER_L"} <= _plot_controls(stats)
    assert "PLOT_P" in _plot_controls(metdata)
    assert "PLOT_RADIATION" not in _plot_controls(metdata)
    assert not {
        "PLOT_FRICTION_VELOCITY",
        "PLOT_REYNOLDS_FLUXES",
        "PLOT_MOISTURE_FLUXES",
        "PLOT_Z_OVER_L",
    } & _plot_controls(tke)


def test_new_drivers_return_without_loading_when_disabled(monkeypatch):
    for module, overrides in (
        (quadrant, {"plot_quadrant_scatter": False, "plot_quadrant_joint_pdf": False}),
        (radiation, {"plot_radiation": False}),
        (fluxes, {"plot_reynolds_fluxes": False, "plot_moisture_fluxes": False}),
        (timescales, {"plot_integral_timescale": False}),
    ):
        monkeypatch.setattr(module, "load_driver_config", lambda path: (_ for _ in ()).throw(AssertionError))
        assert module.run(flag_overrides={**overrides, "save_figures": False}) == []
