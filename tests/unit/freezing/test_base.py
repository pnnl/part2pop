# tests/unit/freezing/test_base.py

from part2pop.freezing.base import retrieve_Jhet_val


def test_retrieve_Jhet_val_for_known_species():
    m, b = retrieve_Jhet_val("SO4", spec_modifications={})
    float(m)
    float(b)
