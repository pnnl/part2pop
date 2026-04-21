import importlib

def test_import_distributions():
    # Smoke test: module should import successfully
    importlib.import_module("part2pop.analysis.distributions")
