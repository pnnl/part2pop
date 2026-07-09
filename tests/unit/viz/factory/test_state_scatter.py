import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from part2pop import build_population
from part2pop.viz.factory.state_scatter import StateScatterPlotter


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


def test_state_scatter_requires_xy():
    with pytest.raises(ValueError):
        StateScatterPlotter({"xvar": "Ddry"})


def test_state_scatter_prep_with_optional_channels():
    plotter = StateScatterPlotter(
        {
            "xvar": "Ddry",
            "yvar": "Dwet",
            "cvar": "mass_dry",
            "svar": "mass_tot",
            "clabel": "custom label",
            "style": {"marker": "x"},
            "colorbar": False,
        }
    )
    prepared = plotter.prep(_population())

    assert prepared["x"].shape == (3,)
    assert prepared["y"].shape == (3,)
    assert prepared["c"].shape == (3,)
    assert prepared["s"].shape == (3,)
    assert np.all(prepared["x"] > 0)
    assert np.all(prepared["y"] > 0)
    assert prepared["xlabel"].startswith("dry diameter")
    assert prepared["ylabel"].startswith("wet diameter")
    assert prepared["clabel"] == "custom label"
    assert prepared["xscale"] == "log"
    assert prepared["yscale"] == "log"


def test_state_scatter_raises_on_length_mismatch():
    plotter = StateScatterPlotter({"xvar": "Ddry", "yvar": "mass_dry"})
    prepared = plotter.prep(_population())
    prepared["y"] = prepared["y"][:1]

    fig, ax = plt.subplots()
    with pytest.raises(ValueError):
        plotter.plot(ax=ax, prepared=prepared)


def test_state_scatter_plot_adds_colorbar():
    plotter = StateScatterPlotter({"xvar": "Ddry", "yvar": "Dwet", "cvar": "mass_dry", "colorbar": True})
    fig, ax = plt.subplots()

    plotter.plot(_population(), ax)

    assert len(fig.axes) == 2
    assert fig.axes[1].get_ylabel() == "dry particle mass [kg]"


def test_state_scatter_plot_accepts_prepared_data():
    plotter = StateScatterPlotter({"xvar": "Ddry", "yvar": "Dwet", "cvar": "mass_dry", "colorbar": False})
    prepared = plotter.prep(_population())
    fig, ax = plt.subplots()

    plotter.plot(ax=ax, prepared=prepared)

    assert len(ax.collections) == 1
    assert len(fig.axes) == 1
    assert ax.get_xlabel()
    assert ax.get_ylabel()
