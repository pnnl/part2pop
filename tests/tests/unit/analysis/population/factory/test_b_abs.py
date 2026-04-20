import importlib

def test_import_b_abs():
    # Smoke test: module should import successfully
    importlib.import_module("part2pop.analysis.population.factory.b_abs")
