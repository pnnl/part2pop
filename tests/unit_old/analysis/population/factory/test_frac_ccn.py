import importlib

def test_import_frac_ccn():
    # Smoke test: module should import successfully
    importlib.import_module("part2pop.analysis.population.factory.frac_ccn")
