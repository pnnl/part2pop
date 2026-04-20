import importlib

def test_import_nucleating_sites():
    # Smoke test: module should import successfully
    importlib.import_module("part2pop.analysis.population.factory.nucleating_sites")
