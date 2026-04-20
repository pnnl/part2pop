import importlib

def test_import_P_frz():
    # Smoke test: module should import successfully
    importlib.import_module("part2pop.analysis.particle.factory.P_frz")
