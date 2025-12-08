import importlib

def test_import_registry():
    # Smoke test: module should import successfully
    importlib.import_module("part2pop.analysis.population.factory.registry")
