import importlib

def test_import___init__():
    # Smoke test: module should import successfully
    importlib.import_module("part2pop.analysis.particle.__init__")
