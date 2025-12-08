import importlib

def test_import_defaults():
    # Smoke test: module should import successfully
    importlib.import_module("part2pop.analysis.defaults")
