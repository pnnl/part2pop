import matplotlib
matplotlib.use("Agg")  # headless backend for CI

import matplotlib.pyplot as plt
import numpy as np
import pytest

from part2pop.viz.factory import state_scatter
from part2pop.viz.factory.state_scatter import StateScatterPlotter


class _DummyVar:
    def __init__(self, name, data):
        self.name = name
        self.data = data
        self.meta = type(
            "Meta",
            (),
            {"long_label": name, "units": "u", "scale": "linear"},
        )

    def compute_all(self, population):
        return self.data

    def compute(self, population):
        return self.data


def _stub_build_variable(population):
    def _build(name, scope_or_cfg="population", var_cfg=None):
        # state_scatter passes the var_cfg dict as the second positional arg;
        # tolerate either (scope, var_cfg) or just (var_cfg,) calling styles.
        data = population[name]
        return _DummyVar(name, data)
    return _build


def test_state_scatter_requires_xy():
    with pytest.raises(ValueError):
        StateScatterPlotter({"xvar": "x-only"})


def test_state_scatter_prep_with_optional_channels(monkeypatch):
    population = {
        "x": np.array([0.0, 1.0]),
        "y": np.array([10.0, 20.0]),
        "c": np.array([5.0, 6.0]),
        "s": np.array([0.5, 0.6]),
    }
    monkeypatch.setattr(state_scatter, "build_variable", _stub_build_variable(population))

    cfg = {
        "xvar": "x",
        "yvar": "y",
        "cvar": "c",
        "svar": "s",
        "clabel": "custom label",
        "style": {"marker": "x"},
        "colorbar": False,
    }
    plotter = StateScatterPlotter(cfg)
    prepared = plotter.prep(population)

    assert prepared["x"].tolist() == [0.0, 1.0]
    assert prepared["y"].tolist() == [10.0, 20.0]
    assert prepared["c"].tolist() == [5.0, 6.0]
    assert prepared["s"].tolist() == [0.5, 0.6]
    assert prepared["xlabel"].startswith("x")
    assert prepared["ylabel"].startswith("y")
    assert prepared["clabel"] == "custom label"
    assert prepared["xscale"] == "linear"
    assert prepared["yscale"] == "linear"


def test_state_scatter_raises_on_length_mismatch(monkeypatch):
    population = {
        "x": np.array([0.0, 1.0]),
        "y": np.array([10.0]),
    }
    monkeypatch.setattr(state_scatter, "build_variable", _stub_build_variable(population))

    plotter = StateScatterPlotter({"xvar": "x", "yvar": "y"})
    with pytest.raises(ValueError):
        plotter.prep(population)


def test_state_scatter_raises_on_color_length_mismatch(monkeypatch):
    population = {
        "x": np.array([0.0, 1.0]),
        "y": np.array([10.0, 20.0]),
        "c": np.array([1.0]),  # wrong length
    }
    monkeypatch.setattr(state_scatter, "build_variable", _stub_build_variable(population))

    plotter = StateScatterPlotter({"xvar": "x", "yvar": "y", "cvar": "c"})
    with pytest.raises(ValueError):
        plotter.prep(population)


def test_state_scatter_raises_on_size_length_mismatch(monkeypatch):
    population = {
        "x": np.array([0.0, 1.0]),
        "y": np.array([10.0, 20.0]),
        "s": np.array([0.1]),  # wrong length
    }
    monkeypatch.setattr(state_scatter, "build_variable", _stub_build_variable(population))

    plotter = StateScatterPlotter({"xvar": "x", "yvar": "y", "svar": "s"})
    with pytest.raises(ValueError):
        plotter.prep(population)


def test_state_scatter_plot_adds_colorbar(monkeypatch):
    population = {
        "x": np.array([0.0, 1.0]),
        "y": np.array([10.0, 20.0]),
        "c": np.array([1.0, 2.0]),
    }
    monkeypatch.setattr(state_scatter, "build_variable", _stub_build_variable(population))

    cfg = {"xvar": "x", "yvar": "y", "cvar": "c", "colorbar": True}
    plotter = StateScatterPlotter(cfg)
    fig, ax = plt.subplots()

    plotter.plot(population, ax)

    # colorbar adds a second axes to the figure
    assert len(fig.axes) == 2
    # and inherits a label from meta when not overridden
    assert fig.axes[1].get_ylabel() == "c [u]"
