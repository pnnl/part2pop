import io

import numpy as np
import pytest

import part2pop.population.utils as utils


class _FakeSpecies:
    def __init__(self, molar_mass):
        self.molar_mass = molar_mass


def test_parse_formula_with_parentheses_and_unknown(monkeypatch):
    tokens = ["Na", "Cl", "SO4", "NH4"]

    counts = utils._parse_formula("(NH4)2SO4", tokens)
    assert counts == {"NH4": 2, "SO4": 1}

    with pytest.raises(ValueError):
        utils._parse_formula("BadToken", tokens)


def test_mass_fraction_from_counts(monkeypatch):
    def fake_get_species(name):
        return _FakeSpecies(molar_mass={"X": 10.0, "Y": 5.0}[name])

    monkeypatch.setattr(utils, "get_species", fake_get_species)
    fracs = utils._mass_fraction_from_counts({"X": 1, "Y": 2})

    assert pytest.approx(fracs["X"]) == 0.5
    assert pytest.approx(fracs["Y"]) == 0.5


def test_expand_compounds_for_population(monkeypatch):
    # stub available tokens and species masses
    monkeypatch.setattr(utils, "_read_available_species_tokens", lambda: ["Na", "Cl", "X"])
    monkeypatch.setattr(
        utils,
        "get_species",
        lambda name: _FakeSpecies({"Na": 23.0, "Cl": 35.0, "X": 10.0}[name]),
    )

    names_list = [["NaCl", "Na", "X2"]]
    fracs_list = [[0.5, 0.25, 0.25]]

    out_names, out_fracs = utils.expand_compounds_for_population(names_list, fracs_list)

    assert len(out_names) == len(out_fracs) == 1
    # NaCl expands to Na + Cl with molar masses; X2 stays as X
    merged = dict(zip(out_names[0], out_fracs[0]))
    # composition fractions normalized
    assert pytest.approx(sum(out_fracs[0])) == 1.0
    assert set(merged) == {"Na", "Cl", "X"}
    # Na appears from NaCl plus explicit Na
    assert merged["Na"] > merged["Cl"]
    assert merged["X"] > 0
