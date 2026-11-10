from io import StringIO

import numpy as np
import pytest

from part2pop import AerosolSpecies, Particle, get_species
from part2pop.freezing import base
from part2pop.freezing.base import retrieve_Jhet_val
from part2pop.freezing.builder import build_freezing_particle


@pytest.mark.parametrize(
    "name,expected",
    [
        ("BC", (18.06, -2.09)),
        ("bC", (18.06, -2.09)),
        ("OC", (72.61, -10.59)),
        ("oin", (54.53, -6.61)),
        ("BIOLOGICAL", (15.48, 4.98)),
        ("MSA", (0.0, 0.0)),
    ],
)
def test_exact_tabulated_names_remain_case_insensitive(name, expected):
    actual = retrieve_Jhet_val(name)
    assert tuple(float(v) for v in actual) == expected


@pytest.mark.parametrize(
    "name", ["B", "O", "ARO", "API", "LIM", "bio", "BC_extra", "unrelated_custom"]
)
def test_prefixes_and_unknown_names_do_not_inherit_table_coefficients(name):
    with pytest.raises(ValueError, match="not found"):
        retrieve_Jhet_val(name)


@pytest.mark.parametrize(
    "name,expected", [("BC", (18.06, -2.09)), ("biological", (15.48, 4.98))]
)
def test_table_whitespace_comments_and_longer_names_do_not_change_exact_match(
    monkeypatch, name, expected
):
    table = "\n # heading\n  BC 18.06 -2.09\nBC_extra 99 88\n biological 15.48 4.98\n"
    monkeypatch.setattr(base, "open_dataset", lambda path: StringIO(table))
    assert tuple(float(v) for v in retrieve_Jhet_val(name)) == expected


@pytest.mark.parametrize(
    "overrides",
    [
        {"m_log10_Jhet": 0.0, "b_log10_Jhet": 0.0},
        {"m_log10_Jhet": 0.0},
        {"b_log10_Jhet": np.nan},
    ],
)
def test_exact_match_keeps_zero_partial_and_nan_overrides(overrides):
    before = overrides.copy()
    actual = retrieve_Jhet_val("BC", overrides)
    expected = (
        overrides.get("m_log10_Jhet", 18.06),
        overrides.get("b_log10_Jhet", -2.09),
    )
    np.testing.assert_equal(np.array(actual, dtype=float), expected)
    assert list(overrides) == list(before)
    np.testing.assert_equal(list(overrides.values()), list(before.values()))


@pytest.mark.parametrize("name", ["B", "ARO", "API", "unrelated_custom"])
def test_custom_material_without_coefficients_has_no_freezing_activity(name):
    custom = AerosolSpecies(name, density=2000.0, kappa=0.1, molar_mass=0.1)
    particle = Particle((custom, get_species("H2O", None)), [1e-18, 1e-17])
    frozen = build_freezing_particle(particle, {"morphology": "homogeneous"})
    assert np.isnan(frozen.m_log10_Jhet[0]) and np.isnan(frozen.b_log10_Jhet[0])
    assert frozen.INSA == 0.0
    assert frozen.compute_Jhet(260.0) == 0.0


@pytest.mark.parametrize(
    "supplied,missing",
    [("m_log10_Jhet", "b_log10_Jhet"), ("b_log10_Jhet", "m_log10_Jhet")],
)
def test_partial_custom_override_cannot_borrow_missing_coefficient(supplied, missing):
    custom = AerosolSpecies("B", density=2000.0, kappa=0.1, molar_mass=0.1)
    particle = Particle((custom, get_species("H2O", None)), [1e-18, 1e-17])
    cfg = {"morphology": "homogeneous", "species_modifications": {"B": {supplied: 0.0}}}
    with pytest.warns(UserWarning, match="missing " + missing):
        frozen = build_freezing_particle(particle, cfg)
    assert frozen.INSA == 0.0
    assert frozen.compute_Jhet(260.0) == 0.0


@pytest.mark.parametrize("name", ["B", "ARO"])
def test_explicit_custom_pair_still_sets_its_rate(name):
    custom = AerosolSpecies(name, density=2000.0, kappa=0.1, molar_mass=0.1)
    particle = Particle((custom, get_species("H2O", None)), [1e-18, 1e-17])
    cfg = {
        "morphology": "homogeneous",
        "species_modifications": {name: {"m_log10_Jhet": 0.0, "b_log10_Jhet": 2.0}},
    }
    frozen = build_freezing_particle(particle, cfg)
    assert frozen.compute_Jhet(260.0) == pytest.approx(100.0)
