import io
from pathlib import Path

import numpy as np
import part2pop
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


def test_parse_formula_parentheses_errors():
    tokens = ["Na", "Cl"]
    with pytest.raises(ValueError):
        utils._parse_formula("(Na", tokens)
    with pytest.raises(ValueError):
        utils._parse_formula("NaCl)", tokens)


def test_expand_compounds_requires_matching_lengths(monkeypatch):
    monkeypatch.setattr(utils, "_read_available_species_tokens", lambda: ["Na"])
    monkeypatch.setattr(utils, "get_species", lambda name: _FakeSpecies({"Na": 23.0}[name]))
    with pytest.raises(ValueError):
        utils.expand_compounds_for_population([["Na"]], [[1.0], [0.5]])


def test_expand_compounds_skips_zero_fractions(monkeypatch):
    monkeypatch.setattr(utils, "_read_available_species_tokens", lambda: ["Na", "Cl"])
    monkeypatch.setattr(
        utils,
        "get_species",
        lambda name: _FakeSpecies({"Na": 23.0, "Cl": 35.0}[name]),
    )
    names_list = [["Na", "Cl"]]
    fracs_list = [[0.0, 1.0]]
    out_names, out_fracs = utils.expand_compounds_for_population(names_list, fracs_list)
    assert out_names == [["Cl"]]
    assert pytest.approx(out_fracs[0][0]) == 1.0


def test_mass_fraction_from_counts_zero_total(monkeypatch):
    monkeypatch.setattr(
        utils,
        "get_species",
        lambda name: _FakeSpecies({"X": 1.0}[name]),
    )
    with pytest.raises(ValueError):
        utils._mass_fraction_from_counts({"X": 0})


def test_normalize_zero_sum():
    assert utils._normalize([0.0, 0.0]) == [0.0, 0.0]


def test_expand_compounds_sublist_mismatch(monkeypatch):
    monkeypatch.setattr(utils, "_read_available_species_tokens", lambda: ["Na"])
    monkeypatch.setattr(utils, "get_species", lambda name: _FakeSpecies({"Na": 23.0}[name]))
    with pytest.raises(ValueError):
        utils.expand_compounds_for_population([["Na"]], [[1.0, 0.0]])


def test_expand_compounds_known_tokens_only(monkeypatch):
    monkeypatch.setattr(utils, "_read_available_species_tokens", lambda: ["Na", "Cl"])
    monkeypatch.setattr(
        utils,
        "get_species",
        lambda name: _FakeSpecies({"Na": 23.0, "Cl": 35.0}[name]),
    )

    names_list = [["Na", "Cl"]]
    fracs_list = [[0.6, 0.4]]
    out_names, out_fracs = utils.expand_compounds_for_population(names_list, fracs_list)

    assert out_names == [["Na", "Cl"]]
    assert pytest.approx(out_fracs[0][0] + out_fracs[0][1]) == 1.0


def test_parse_formula_trailing_characters(monkeypatch):
    tokens = ["Na", "Cl"]
    with pytest.raises(ValueError, match="Unknown token"):
        utils._parse_formula("NaClX", tokens)


def test_read_available_species_tokens(monkeypatch):
    data_root = Path(part2pop.__file__).resolve().parent / "data" / "species_data"
    data_file = data_root / "aero_data.dat"

    def fake_open_dataset(name):
        return open(data_file, "r", encoding="utf-8")

    monkeypatch.setattr(utils, "open_dataset", fake_open_dataset)
    tokens = utils._read_available_species_tokens()
    assert "SO4" in tokens
    assert all(isinstance(tok, str) for tok in tokens)


def test_parse_formula_handles_nested_groups_and_multiplier():
    tokens = ["NH4", "SO4", "NO3"]
    counts = utils._parse_formula("(NH4)12SO4NO3", tokens)
    assert counts == {"NH4": 12, "SO4": 1, "NO3": 1}


def test_expand_compounds_merges_parsed_tokens(monkeypatch):
    monkeypatch.setattr(utils, "_read_available_species_tokens", lambda: ["Na", "Cl", "SO4", "NH4"])
    molar = {"Na": 23.0, "Cl": 35.0, "SO4": 96.0, "NH4": 18.0}
    monkeypatch.setattr(utils, "get_species", lambda name: _FakeSpecies(molar[name]))

    names_list = [["(NH4)2SO4", "NaCl"], ["Na"]]
    fracs_list = [[0.4, 0.6], [1.0]]

    out_names, out_fracs = utils.expand_compounds_for_population(names_list, fracs_list)
    assert set(out_names[0]) == {"NH4", "SO4", "Na", "Cl"}
    assert pytest.approx(sum(out_fracs[0])) == 1.0
    assert set(out_names[1]) == {"Na"}


def test_expand_compounds_normalizes_with_duplicate_known_tokens(monkeypatch):
    monkeypatch.setattr(utils, "_read_available_species_tokens", lambda: ["X"])
    monkeypatch.setattr(utils, "get_species", lambda name: _FakeSpecies({"X": 10.0}[name]))
    names_list = [["X", "X", "X"]]
    fracs_list = [[0.2, 0.3, 0.5]]
    out_names, out_fracs = utils.expand_compounds_for_population(names_list, fracs_list)
    assert out_names[0] == ["X"]
    assert pytest.approx(out_fracs[0][0]) == 1.0
