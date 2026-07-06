import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from part2pop import build_population
from part2pop.viz.factory.state_line import StateLinePlotter


def _population():
    return build_population(
        {
            "type": "monodisperse",
            "aero_spec_names": [["SO4"], ["SO4"], ["SO4"]],
            "N": [1.0e5, 2.0e5, 3.0e5],
            "D": [50e-9, 100e-9, 200e-9],
            "aero_spec_fracs": [[1.0], [1.0], [1.0]],
        }
    )


def _dnd_config():
    edges = np.asarray([1.0e-8, 8.0e-8, 1.5e-7, 3.0e-7])
    return {
        "varname": "dNdlnD",
        "var_cfg": {
            "method": "hist",
            "edges": edges,
            "diam_grid": np.sqrt(edges[:-1] * edges[1:]),
        },
    }


def test_state_line_requires_varname():
    with pytest.raises(ValueError):
        StateLinePlotter({})


def test_state_line_prep_for_dndln_d():
    plotter = StateLinePlotter(_dnd_config())
    prepared = plotter.prep(_population())

    assert prepared["x"].shape == (3,)
    assert prepared["y"].shape == (3,)
    assert prepared["xlabel"].startswith("dry diameter")
    assert "number size distribution" in prepared["ylabel"].lower()
    assert prepared["xscale"] == "log"
    assert prepared["yscale"] == "linear"


def test_state_line_plot_sets_labels_and_limits():
    plotter = StateLinePlotter(_dnd_config())
    fig, ax = plt.subplots()
    plotter.plot(_population(), ax)

    assert len(ax.lines) == 1
    assert ax.get_xlabel()
    assert ax.get_ylabel()
    x_limits = ax.get_xlim()
    assert x_limits[0] <= min(plotter.prep(_population())["x"])
    assert x_limits[1] >= max(plotter.prep(_population())["x"])


def test_state_line_plot_accepts_prepared_data():
    plotter = StateLinePlotter(_dnd_config())
    prepared = plotter.prep(_population())
    fig, ax = plt.subplots()

    plotter.plot(ax=ax, prepared=prepared)

    assert len(ax.lines) == 1
    assert ax.get_xlabel()
    assert ax.get_ylabel()


def test_state_line_plot_prepared():
    plotter = StateLinePlotter(_dnd_config())
    prepared = plotter.prep(_population())
    fig, ax = plt.subplots()

    plotter.plot_prepared(prepared, ax)

    assert len(ax.lines) == 1


def test_state_line_raises_on_conflicting_axes():
    plotter = StateLinePlotter(
        {"varname": "b_ext", "var_cfg": {"wvl_grid": [550e-9, 650e-9], "rh_grid": [0.4, 0.6]}}
    )

    with pytest.raises(ValueError, match="one varying axis"):
        plotter.prep(_population())


def test_state_line_raises_when_no_axis_available():
    plotter = StateLinePlotter(
        {"varname": "b_ext", "var_cfg": {"wvl_grid": [550e-9], "rh_grid": [0.5]}}
    )

    with pytest.raises(ValueError, match="single wavelength and single RH"):
        plotter.prep(_population())


def test_state_line_scalar_outputs_are_supported():
    plotter = StateLinePlotter({"varname": "Nccn", "var_cfg": {"s_grid": [1.0]}})
    prepared = plotter.prep(_population())
    fig, ax = plt.subplots()

    plotter.plot(ax=ax, prepared=prepared)

    assert prepared["x"].tolist() == [1.0]
    assert prepared["y"].shape == (1,)
    assert len(ax.lines) == 1


def test_state_line_rejects_unsupported_variable():
    plotter = StateLinePlotter({"varname": "unknown"})

    with pytest.raises(ValueError, match="does not support"):
        plotter.prep(_population())
