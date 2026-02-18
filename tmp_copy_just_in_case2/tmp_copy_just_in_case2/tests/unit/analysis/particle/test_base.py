import importlib

def test_import_base():
    # Smoke test: module should import successfully
    importlib.import_module("part2pop.analysis.particle.base")
