import numpy as np

from part2pop.analysis import defaults as defs


def test_get_defaults_handles_aliases_and_missing():
    base = defs.get_defaults_for_variable("Nccn")
    lower = defs.get_defaults_for_variable("nccn")
    assert base == lower
    assert "s_grid" in base and "s_eval" in base

    missing = defs.get_defaults_for_variable("unknown_variable")
    assert missing == defs.get_defaults_for_variable("__fallback__")
    # Returned dicts are copies
    missing["x"] = 1
    assert "x" not in defs.get_defaults_for_variable("unknown_variable")


def test_all_defaults_returns_copy():
    all_defs = defs.all_defaults()
    assert "__fallback__" in all_defs
    # Mutating returned dict does not mutate source
    all_defs["Nccn"]["extra"] = 123
    assert "extra" not in defs.get_defaults_for_variable("Nccn")
