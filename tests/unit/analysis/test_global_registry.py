import importlib

def test_import_global_registry():
    # Smoke test: module should import successfully
    importlib.import_module("part2pop.analysis.global_registry")
