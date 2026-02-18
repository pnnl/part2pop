import importlib

def test_import_Dwet():
    # Smoke test: module should import successfully
    importlib.import_module("part2pop.analysis.particle.factory.Dwet")
