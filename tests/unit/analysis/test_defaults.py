# tests/unit/analysis/test_defaults.py

from pyparticle.analysis.defaults import get_defaults_for_variable, all_defaults


def test_get_defaults_for_known_variable():
    d = get_defaults_for_variable("dNdlnD")
    # Should at least contain grid information
    assert "D_min" in d
    assert "D_max" in d
    assert "nbins" in d


def test_get_defaults_for_unknown_returns_fallback():
    d = get_defaults_for_variable("this_is_not_real")
    # Fallback may be empty, but should be a dict
    assert isinstance(d, dict)


def test_all_defaults_contains_known_key():
    d = all_defaults()
    assert "dNdlnD" in d
