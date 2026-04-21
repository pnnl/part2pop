import importlib
import sys
import types


def _reload_constants(monkeypatch, fake_constants=None):
    sys.modules.pop("part2pop.constants", None)
    fake_package = types.ModuleType("scipy")
    if fake_constants is None:
        fake_constants = types.ModuleType("scipy.constants")
    fake_package.constants = fake_constants
    monkeypatch.setitem(sys.modules, "scipy", fake_package)
    monkeypatch.setitem(sys.modules, "scipy.constants", fake_constants)
    return importlib.import_module("part2pop.constants")


def test_constants_fallback_when_scipy_missing(monkeypatch):
    consts = _reload_constants(monkeypatch, fake_constants=types.ModuleType("scipy.constants"))
    assert consts.MOLAR_MASS_DRY_AIR > 0
    assert consts.DENSITY_LIQUID_WATER > 0
    assert consts.R == 8.31446261815324


def test_constants_uses_scipy_values(monkeypatch):
    fake_constants = types.ModuleType("scipy.constants")
    fake_constants.R = 9.5
    consts = _reload_constants(monkeypatch, fake_constants=fake_constants)
    assert consts.R == 9.5
