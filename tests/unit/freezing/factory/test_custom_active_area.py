import numpy as np
import pytest

from part2pop import AerosolSpecies, Particle, ParticlePopulation, get_species
from part2pop.analysis.population.factory.INSA_distribution import build as build_insa
from part2pop.freezing.builder import build_freezing_particle, build_freezing_population


@pytest.mark.parametrize("method", ["hist", "kde"])
@pytest.mark.parametrize("source", ["population", "config", "none", "override"])
def test_insa_distribution_uses_authoritative_freezing_modifications(method, source):
    custom = AerosolSpecies(
        "custom_distribution", density=2000.0, kappa=0.1, molar_mass=0.1
    )
    particle = Particle((custom, get_species("H2O", None)), [1e-18, 1e-17])
    active = {custom.name: {"m_log10_Jhet": 0.0, "b_log10_Jhet": 1.0}}
    inactive = {custom.name: {"m_log10_Jhet": np.nan, "b_log10_Jhet": np.nan}}
    inherited = inactive if source == "override" else active
    pop = ParticlePopulation(
        particle.species,
        np.array([particle.masses]),
        np.array([100.0]),
        [0],
        species_modifications=inherited,
    )
    area = 4 * np.pi * (3 * (1e-18 / 2000.0) / (4 * np.pi)) ** (2 / 3)
    cfg = {"method": method, "insa_grid": np.geomspace(area / 10, area * 10, 51)}
    if source in ("config", "override"):
        cfg["species_modifications"] = active
        if source == "config":
            pop.species_modifications = {}
    elif source == "none":
        cfg["species_modifications"] = None
    before_modifications = pop.species_modifications
    out = build_insa(cfg).compute(pop, as_dict=True)
    assert np.count_nonzero(out["dNdlnINSA"]) > 0
    if method == "hist":
        count = np.sum(out["dNdlnINSA"] * np.diff(np.log(out["edges"])))
        assert count == pytest.approx(100.0)
    assert pop.species_modifications is before_modifications
    if source == "override":
        assert np.isnan(pop.species_modifications[custom.name]["m_log10_Jhet"])


def test_explicit_empty_distribution_modifications_disable_population_defaults():
    custom = AerosolSpecies(
        "custom_disabled", density=2000.0, kappa=0.1, molar_mass=0.1
    )
    particle = Particle((custom, get_species("H2O", None)), [1e-18, 1e-17])
    pop = ParticlePopulation(
        particle.species,
        np.array([particle.masses]),
        np.array([100.0]),
        [0],
        species_modifications={custom.name: {"m_log10_Jhet": 0.0, "b_log10_Jhet": 1.0}},
    )
    out = build_insa({"method": "hist", "species_modifications": {}}).compute(pop)
    np.testing.assert_array_equal(out, np.zeros_like(out))


@pytest.mark.parametrize("intercept", [0.0, 16.0])
def test_custom_coefficients_create_analytic_active_area(intercept):
    s = AerosolSpecies("custom_ice", density=2000.0, kappa=0.1, molar_mass=0.1)
    p = Particle((s, get_species("H2O", None)), [1e-18, 1e-17])
    pop = ParticlePopulation(p.species, np.array([p.masses]), np.array([1.0]), [0])
    cfg = {
        "morphology": "homogeneous",
        "species_modifications": {
            "custom_ice": {"m_log10_Jhet": 0.0, "b_log10_Jhet": intercept}
        },
    }
    f = build_freezing_population(pop, cfg)
    area = 4 * np.pi * (3 * (1e-18 / 2000.0) / (4 * np.pi)) ** (2 / 3)
    assert f.INSA[0] == pytest.approx(area, rel=1e-12, abs=0.0)
    assert f.compute_Jhets(260.0, cfg)[0] == pytest.approx(10.0**intercept, rel=1e-12)
    expected = 1 - np.exp(-(10.0**intercept) * area)
    assert f.get_freezing_probs(260.0, cfg)[0] == pytest.approx(
        expected, rel=1e-12, abs=0.0
    )


@pytest.mark.parametrize(
    "supplied,missing",
    [("m_log10_Jhet", "b_log10_Jhet"), ("b_log10_Jhet", "m_log10_Jhet")],
)
def test_custom_partial_pair_keeps_existing_warning_and_zero_area(supplied, missing):
    s = AerosolSpecies("custom_partial", density=2000.0, kappa=0.1, molar_mass=0.1)
    p = Particle((s, get_species("H2O", None)), [1e-18, 1e-17])
    cfg = {
        "morphology": "homogeneous",
        "species_modifications": {s.name: {supplied: 0.0}},
    }
    with pytest.warns(UserWarning, match="missing " + missing):
        f = build_freezing_particle(p, cfg)
    assert f.INSA == 0.0


def test_custom_zero_mass_has_no_area():
    s = AerosolSpecies("custom_zero", density=2000.0, kappa=0.1, molar_mass=0.1)
    p = Particle((s, get_species("H2O", None)), [0.0, 1e-17])
    f = build_freezing_particle(
        p,
        {
            "morphology": "homogeneous",
            "species_modifications": {
                s.name: {"m_log10_Jhet": 0.0, "b_log10_Jhet": 0.0}
            },
        },
    )
    assert f.INSA == 0.0


def test_custom_area_fix_preserves_existing_water_activity_classification():
    custom = AerosolSpecies(
        "custom_with_solute", density=2000.0, kappa=0.1, molar_mass=0.1
    )
    p = Particle(
        (custom, get_species("SO4", None), get_species("H2O", None)),
        [1e-18, 0.096e-17, 1e-17],
    )
    cfg = {
        "morphology": "homogeneous",
        "species_modifications": {
            custom.name: {"m_log10_Jhet": 0.0, "b_log10_Jhet": 0.0}
        },
    }
    f = build_freezing_particle(p, cfg)
    assert f.get_aw() == pytest.approx(np.exp(-0.01801528), rel=1e-12, abs=0.0)


def test_custom_and_tabulated_material_share_equivalent_sphere_area():
    custom = AerosolSpecies(
        "mixed_custom_ice", density=2000.0, kappa=0.1, molar_mass=0.1
    )
    specs = (
        custom,
        get_species("BC", None),
        get_species("SO4", None),
        get_species("H2O", None),
    )
    p = Particle(
        specs, np.array([2e-21, 1e-21, 8e-21, 20e-21]) * [s.density for s in specs]
    )
    cfg = {
        "morphology": "homogeneous",
        "species_modifications": {
            custom.name: {"m_log10_Jhet": 0.0, "b_log10_Jhet": 0.0}
        },
    }
    f = build_freezing_particle(p, cfg)
    equivalent_sphere_area = (36 * np.pi * (3e-21) ** 2) ** (1 / 3)
    assert f.INSA == pytest.approx(equivalent_sphere_area, rel=1e-12, abs=0.0)
    # The retained morphology uses one sphere, rather than the sum of separate areas.
    separate_areas = (36 * np.pi * (2e-21) ** 2) ** (1 / 3) + (
        36 * np.pi * (1e-21) ** 2
    ) ** (1 / 3)
    assert f.INSA < separate_areas


def test_custom_active_salt_uses_its_own_volume_with_prefix_named_species():
    na, nacl, water = (get_species(name, None) for name in ("Na", "NaCl", "H2O"))
    p = Particle((na, nacl, water), [2.3e-19, 5.85e-19, 1e-17])
    cfg = {
        "morphology": "homogeneous",
        "species_modifications": {"NaCl": {"m_log10_Jhet": 0.0, "b_log10_Jhet": 0.0}},
    }
    f = build_freezing_particle(p, cfg)
    expected = 4 * np.pi * (3 * p.masses[1] / nacl.density / (4 * np.pi)) ** (2 / 3)
    assert f.INSA == pytest.approx(expected, rel=1e-12, abs=0.0)
