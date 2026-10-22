import numpy as np
import pytest
from part2pop.freezing.builder import build_freezing_population

from part2pop import ParticlePopulation, make_particle


def freezing(names, fractions, temperature=250.0):
    part = make_particle(100e-9, names, fractions)
    pop = ParticlePopulation(
        part.species, np.array([part.masses]), np.array([1.0]), [0]
    )
    cfg = {"morphology": "homogeneous"}
    return build_freezing_population(pop, cfg), cfg


@pytest.mark.parametrize(
    "names,fractions",
    [
        (["SO4", "BC", "H2O"], [0.1, 0.0, 0.9]),
        (["SO4", "BC", "OIN", "H2O"], [0.1, 0.0, 0.0, 0.9]),
    ],
)
def test_absent_active_species_gives_zero_rate_and_finite_probability(names, fractions):
    pop, cfg = freezing(names, fractions)
    control, _ = freezing(["SO4", "H2O"], [0.1, 0.9])
    assert pop.compute_Jhets(250.0, cfg)[0] == 0.0
    np.testing.assert_allclose(
        pop.get_freezing_probs(250.0, cfg),
        control.get_freezing_probs(250.0, cfg),
        rtol=1e-12,
        atol=0.0,
    )
    assert np.isfinite(pop.get_freezing_probs(250.0, cfg)).all()


def test_present_active_material_retains_rate():
    pop, cfg = freezing(["BC", "H2O"], [0.1, 0.9])
    part = pop.get_freezing_particle(0, cfg)
    from part2pop.freezing.factory.utils import calculate_Psat

    liquid, ice = calculate_Psat(250.0)
    expected = 10.0 ** (18.06 * (part.get_aw() - ice / liquid) - 2.09)
    assert pop.compute_Jhets(250.0, cfg)[0] == pytest.approx(
        expected, rel=1e-12, abs=0.0
    )


def test_externally_mixed_population_keeps_every_probability_finite():
    first = make_particle(100e-9, ["SO4", "BC", "H2O"], [0.1, 0.0, 0.9])
    second = make_particle(100e-9, ["SO4", "BC", "H2O"], [0.0, 0.1, 0.9])
    pop = ParticlePopulation(
        first.species,
        np.array([first.masses, second.masses]),
        np.array([1.0, 2.0]),
        [0, 1],
    )
    cfg = {"morphology": "homogeneous"}
    frozen = build_freezing_population(pop, cfg)
    rates = frozen.compute_Jhets(250.0, cfg)
    assert rates[0] == 0.0
    assert rates[1] > 0.0
    assert np.isfinite(frozen.get_freezing_probs(250.0, cfg)).all()


def test_two_active_materials_use_volume_not_mass_and_ignore_solute():
    from part2pop import Particle, get_species
    from part2pop.freezing.builder import build_freezing_particle

    specs = tuple(get_species(n, None) for n in ["BC", "OIN", "SO4", "H2O"])
    # BC and OIN occupy 2:1 volume, with different densities; sulfate is inactive.
    masses = np.array([2e-21, 1e-21, 9e-21, 20e-21]) * [s.density for s in specs]
    p = Particle(specs, masses)
    cfg = {
        "morphology": "homogeneous",
        "species_modifications": {
            "BC": {"m_log10_Jhet": 0.0, "b_log10_Jhet": 2.0},
            "OIN": {"m_log10_Jhet": 0.0, "b_log10_Jhet": 3.0},
        },
    }
    f = build_freezing_particle(p, cfg)
    assert f.compute_Jhet(250.0) == pytest.approx(400.0, rel=1e-12)


def test_absent_material_is_not_evaluated_even_if_its_rate_would_overflow():
    from part2pop.freezing.builder import build_freezing_particle

    p = make_particle(100e-9, ["BC", "OIN", "H2O"], [0.0, 0.1, 0.9])
    f = build_freezing_particle(
        p,
        {
            "morphology": "homogeneous",
            "species_modifications": {
                "BC": {"m_log10_Jhet": 0.0, "b_log10_Jhet": 400.0},
                "OIN": {"m_log10_Jhet": 0.0, "b_log10_Jhet": 2.0},
            },
        },
    )
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        assert f.compute_Jhet(250.0) == pytest.approx(100.0)


@pytest.mark.parametrize("names", [["ARO1", "SO4", "H2O"], ["BC", "SO4", "H2O"]])
@pytest.mark.parametrize("temperature", [273.15, 283.15, 298.15])
def test_no_heterogeneous_rate_without_ice_supersaturation(names, temperature):
    # At and above 0 degC a_w <= a_w,ice, so ice cannot nucleate. The packaged
    # SOA rows have m_log10_Jhet = 0, which used to give a constant 10**8.98
    # m^-2 s^-1 at any temperature.
    pop, cfg = freezing(names, [0.05, 0.05, 0.9])
    assert pop.compute_Jhets(temperature, cfg)[0] == 0.0


def test_supercooled_rates_are_unchanged():
    pop, cfg = freezing(["ARO1", "SO4", "H2O"], [0.05, 0.05, 0.9])
    # m = 0 material keeps its constant tabulated rate once supercooled.
    assert pop.compute_Jhets(250.0, cfg)[0] == pytest.approx(10**8.98, rel=1e-12)
