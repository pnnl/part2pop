import builtins
import importlib
import sys


def test_constants_import_and_defaults(monkeypatch):
    # Force a fresh import and require the scipy.constants import to fail
    sys.modules.pop("part2pop.constants", None)

    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "scipy.constants":
            raise ImportError("scipy.constants missing")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    consts = importlib.import_module("part2pop.constants")
    assert consts.MOLAR_MASS_DRY_AIR > 0
    assert consts.DENSITY_LIQUID_WATER > 0
    assert isinstance(consts.R, float)
    assert consts.R == 8.31446261815324
