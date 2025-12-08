import importlib

def test_import_diam_grid():
    # Smoke test: module should import successfully
    importlib.import_module("part2pop.analysis.population.factory.diam_grid")
