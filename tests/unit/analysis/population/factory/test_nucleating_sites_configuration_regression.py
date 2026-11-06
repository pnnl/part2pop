import numpy as np
import pytest

from part2pop import make_particle
from part2pop.population.base import ParticlePopulation
from part2pop.analysis.population.factory.nucleating_sites import build


def _population(water=0.5):
    particle = make_particle(100e-9, ["OIN", "H2O"], [1.0 - water, water])
    mods = {"OIN": {"m_log10_Jhet": 0.0, "b_log10_Jhet": 3.0}}
    return ParticlePopulation(tuple(particle.species), np.array([particle.masses]),
                              np.array([1.0]), [0], species_modifications=mods)


def test_omitted_mapping_inherits_population_freezing_parameters():
    pop = _population()
    cfg = {"T_grid": [260.0, 250.0], "cooling_rate": 1.0}
    inherited = build(cfg).compute(pop)
    explicit = build(dict(cfg, species_modifications=pop.species_modifications)).compute(pop)
    np.testing.assert_allclose(inherited, [0.0, 10000.0], rtol=1e-12, atol=0.0)
    np.testing.assert_array_equal(inherited, explicit)


def test_explicit_empty_mapping_uses_defaults_and_override_replaces_population():
    pop = _population()
    cfg = {"T_grid": [260.0, 250.0], "cooling_rate": 1.0}
    reset = build(dict(cfg, species_modifications={})).compute(pop)
    defaults = _population()
    defaults.species_modifications = {}
    np.testing.assert_array_equal(reset, build(cfg).compute(defaults))
    override = {"OIN": {"m_log10_Jhet": 0.0, "b_log10_Jhet": 2.0}}
    result = build(dict(cfg, species_modifications=override)).compute(pop)
    np.testing.assert_allclose(result, [0.0, 1000.0], rtol=1e-12, atol=0.0)
    assert pop.species_modifications["OIN"]["b_log10_Jhet"] == 3.0


def test_dry_population_is_not_equilibrated_implicitly():
    result = build({"T_grid": [260.0, 250.0], "cooling_rate": 1.0}).compute(_population(water=0.0))
    np.testing.assert_array_equal(result, [0.0, 0.0])


def test_explicit_none_inherits_like_an_omitted_mapping():
    # None used to fall through 'or {}' and reset the population's mapping,
    # unlike the other freezing analyses, which treat None as "inherit".
    pop = _population()
    cfg = {"T_grid": [260.0, 250.0], "cooling_rate": 1.0}
    inherited = build(cfg).compute(pop)
    with_none = build(dict(cfg, species_modifications=None)).compute(pop)
    np.testing.assert_array_equal(with_none, inherited)
    np.testing.assert_allclose(with_none, [0.0, 10000.0], rtol=1e-12, atol=0.0)
