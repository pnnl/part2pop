import importlib

def test_import_spec_mass_conc():
    # Smoke test: module should import successfully
    importlib.import_module("part2pop.analysis.population.factory.spec_mass_conc")
