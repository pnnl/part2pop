import matplotlib
matplotlib.use("Agg")  # headless backend for tests

import numpy as np
import matplotlib.pyplot as plt
import pytest

from part2pop.viz.factory import state_line
from part2pop.viz.factory.state_line import StateLinePlotter


class _DummyVar:
    def __init__(self, name, data):
        self.name = name
        self.data = data
        self.meta = type(
            "Meta",
            (),
            {"long_label": name, "units": "u", "scale": "linear"},
        )

    def compute(self, population):
        return self.data


def _stub_build_variable(population):
    def _build(name, scope="population", var_cfg=None):
        return _DummyVar(name, population[name])
    return _build


def test_state_line_requires_varname():
    with pytest.raises(ValueError):
        StateLinePlotter({})


def test_state_line_prep_for_dndln_d(monkeypatch):
    population = {
        "dNdlnD": np.array([1.0, 2.0]),
        "diam_grid": np.array([0.1, 0.2]),
    }
    monkeypatch.setattr(state_line, "build_variable", _stub_build_variable(population))

    plotter = StateLinePlotter({"varname": "dNdlnD", "var_cfg": {}})
    prepared = plotter.prep(population)

    assert prepared["x"].tolist() == [0.1, 0.2]
    assert prepared["y"].tolist() == [1.0, 2.0]
    assert prepared["xlabel"].startswith("diam_grid")
    assert prepared["ylabel"].startswith("dNdlnD")
    assert prepared["xscale"] == "linear"
    assert prepared["yscale"] == "linear"


def test_state_line_plot_sets_labels_and_limits(monkeypatch):
    population = {
        "dNdlnD": np.array([3.0, 4.0]),
        "diam_grid": np.array([0.3, 0.4]),
    }
    monkeypatch.setattr(state_line, "build_variable", _stub_build_variable(population))

    plotter = StateLinePlotter({"varname": "dNdlnD"})
    fig, ax = plt.subplots()
    plotter.plot(population, ax)

    assert ax.get_xlabel()
    assert ax.get_ylabel()
    x_limits = ax.get_xlim()
    assert x_limits[0] <= 0.3 and x_limits[1] >= 0.4


def test_state_line_raises_on_conflicting_axes(monkeypatch):
    population = {
        "b_ext": np.array([1.0, 2.0]),
        "wvl_grid": np.array([550.0, 650.0]),
        "rh_grid": np.array([0.4, 0.6]),
    }
    monkeypatch.setattr(state_line, "build_variable", _stub_build_variable(population))

    plotter = StateLinePlotter(
        {"varname": "b_ext", "var_cfg": {"wvl_grid": population["wvl_grid"], "rh_grid": population["rh_grid"]}}
    )

    with pytest.raises(ValueError):
        plotter.prep(population)


def test_state_line_raises_when_no_axis_available(monkeypatch):
    population = {
        "b_ext": np.array([1.0]),
        "wvl_grid": np.array([550.0]),
        "rh_grid": np.array([0.5]),
    }
    monkeypatch.setattr(state_line, "build_variable", _stub_build_variable(population))

    plotter = StateLinePlotter({"varname": "b_ext", "var_cfg": {"wvl_grid": population["wvl_grid"], "rh_grid": population["rh_grid"]}})
    with pytest.raises(ValueError):
        plotter.prep(population)


def test_state_line_scalar_outputs_currently_error(monkeypatch):
    population = {
        "Nccn": np.array([42.0]),
        "s_grid": np.array([0.1]),
    }
    monkeypatch.setattr(state_line, "build_variable", _stub_build_variable(population))

    plotter = StateLinePlotter({"varname": "Nccn"})
    with pytest.raises(TypeError):
        plotter.prep(population)
