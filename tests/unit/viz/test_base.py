import pytest

from part2pop.viz.base import Plotter


def test_plotter_methods_raise_not_implemented():
    plotter = Plotter(type="dummy", config={})

    with pytest.raises(NotImplementedError):
        plotter.prep(object())

    with pytest.raises(NotImplementedError):
        plotter.plot(object(), object())
