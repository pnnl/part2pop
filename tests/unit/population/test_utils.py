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


def test_mass_fraction_from_counts_requires_positive_total(monkeypatch):
    monkeypatch.setattr(utils, "get_species", lambda name: _FakeSpecies(molar_mass=0.0))

    with pytest.raises(ValueError):
        utils._mass_fraction_from_counts({"X": 1})


def test_parse_formula_supports_counts_and_errors():
    tokens = ["SO4", "Na", "Cl"]

    counts = utils._parse_formula("Na2ClSO4", tokens)
    assert counts == {"Na": 2, "Cl": 1, "SO4": 1}

    with pytest.raises(ValueError):
        utils._parse_formula("(Na2", tokens)

    with pytest.raises(ValueError):
        utils._parse_formula("Na2Cl)X", tokens)

    with pytest.raises(ValueError):
        utils._parse_formula("Unknown", tokens)


def test_normalize_handles_zero_sum():
    fracs = utils._normalize([0.0, 0.0, 0.0])
    assert fracs == [0.0, 0.0, 0.0]

    normalized = utils._normalize([1.0, 1.0])
    assert normalized == [0.5, 0.5]


def test_expand_compounds_errors_and_zero_fracs(monkeypatch):
    monkeypatch.setattr(utils, "_read_available_species_tokens", lambda: ["Na", "Cl", "SO4"])
    monkeypatch.setattr(
        utils,
        "get_species",
        lambda name: _FakeSpecies(
            molar_mass={"Na": 23.0, "Cl": 35.0, "SO4": 96.0}[name]
        ),
    )

    # Length mismatch between names_list and fracs_list
    with pytest.raises(ValueError):
        utils.expand_compounds_for_population([["Na"]], [[0.5], [0.5]])

    # Sublist length mismatch
    with pytest.raises(ValueError):
        utils.expand_compounds_for_population([["Na", "Cl"]], [[0.5]])

    # Zero fractions are ignored; normalized result should be identical to a single entry
    names_list = [["Na", "Cl"]]
    fracs_list = [[0.0, 1.0]]
    out_names, out_fracs = utils.expand_compounds_for_population(names_list, fracs_list)
    assert out_names == [["Cl"]]
    assert pytest.approx(out_fracs[0][0]) == 1.0


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


def test_read_available_species_tokens(monkeypatch):
    fake_data = "\n".join(
        [
            "# header",
            "X 100 0 10d-3 0.5",
            "ABC 100 0 20d-3 0.1",
        ]
    )
    monkeypatch.setattr(
        utils,
        "open_dataset",
        lambda _: io.StringIO(fake_data),
    )
    tokens = utils._read_available_species_tokens()
    assert tokens[0] == "ABC"  # sorted longest-first
    assert "X" in tokens


def test_parse_formula_flags_trailing_chars():
    tokens = ["SO4", "Na", "Cl"]
    with pytest.raises(ValueError):
        utils._parse_formula("SO4X", tokens)


def test_mass_fraction_from_counts_with_real_species():
    fracs = utils._mass_fraction_from_counts({"SO4": 1, "BC": 1})
    assert pytest.approx(sum(fracs.values())) == 1.0
    assert fracs["SO4"] > 0.0 and fracs["BC"] > 0.0
