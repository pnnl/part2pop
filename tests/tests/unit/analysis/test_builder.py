import importlib

def test_import_builder():
    # Smoke test: module should import successfully
    importlib.import_module("part2pop.analysis.builder")
