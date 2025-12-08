import importlib

def test_import_avg_Jhet():
    # Smoke test: module should import successfully
    importlib.import_module("part2pop.analysis.population.factory.avg_Jhet")
