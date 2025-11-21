import pytest

from pyparticle.viz.factory.registry import discover_plotter_types


def test_discover_plotter_types_returns_dict():
    types = discover_plotter_types()
    assert isinstance(types, dict)
    for k, v in types.items():
        assert callable(v)
