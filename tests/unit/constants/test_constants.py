import importlib
import sys
import types


def test_constants_import_and_defaults(monkeypatch):
    # Import fresh to hit fallback path when scipy.constants.R missing
    monkeypatch.setitem(importlib.import_module("sys").modules, "scipy.constants", None)
    consts = importlib.import_module("part2pop.constants")
    assert consts.MOLAR_MASS_DRY_AIR > 0
    assert consts.DENSITY_LIQUID_WATER > 0
    assert isinstance(consts.R, float)


def test_constants_uses_scipy_values(monkeypatch):
    fake_constants = types.SimpleNamespace(R=9.5)
    monkeypatch.setitem(sys.modules, "scipy.constants", fake_constants)
    consts = importlib.reload(importlib.import_module("part2pop.constants"))
    assert consts.R == 9.5
