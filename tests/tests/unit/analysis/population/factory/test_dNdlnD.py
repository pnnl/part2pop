import importlib

def test_import_dNdlnD():
    # Smoke test: module should import successfully
    importlib.import_module("part2pop.analysis.population.factory.dNdlnD")
