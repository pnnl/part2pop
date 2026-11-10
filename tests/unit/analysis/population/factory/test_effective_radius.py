import numpy as np
import pytest

from part2pop import ParticlePopulation, make_particle
from part2pop.analysis import build_variable
from part2pop.analysis.population.factory import effective_radius


def pop(diameters, weights):
    parts = [make_particle(d, ["SO4"], [1.0]) for d in diameters]
    return ParticlePopulation(
        parts[0].species,
        np.array([p.masses for p in parts]),
        np.array(weights),
        list(range(len(parts))),
    )


def r_eff(population):
    return effective_radius.build({}).compute(population)


def test_effective_radius_is_not_number_mean_radius():
    assert r_eff(pop([2e-6, 6e-6], [1.0, 1.0])) == pytest.approx(
        2.8e-6, rel=1e-12, abs=0.0
    )


def test_unequal_concentrations_and_scale_invariance():
    p = pop([2e-6, 6e-6], [3.0, 1.0])
    assert r_eff(p) == pytest.approx(2.5e-6, rel=1e-12, abs=0.0)
    p.num_concs *= 100.0
    assert r_eff(p) == pytest.approx(2.5e-6, rel=1e-12, abs=0.0)
    assert r_eff(pop([2e-6], [7.0])) == pytest.approx(
        1e-6, rel=1e-12, abs=0.0
    )


def test_zero_total_area_is_undefined():
    assert np.isnan(r_eff(pop([2e-6], [0.0])))
    empty = ParticlePopulation((), np.empty((0, 0)), np.array([]), [])
    assert np.isnan(r_eff(empty))


def test_effective_radius_matches_wet_volume_to_area_and_ignores_zero_count():
    parts = [
        make_particle(d, ["SO4", "H2O"], fractions)
        for d, fractions in [
            (2e-6, [0.5, 0.5]),
            (6e-6, [0.1, 0.9]),
            (20e-6, [0.5, 0.5]),
        ]
    ]
    weights = np.array([3.0, 1.0, 0.0])
    p = ParticlePopulation(
        parts[0].species, np.array([q.masses for q in parts]), weights, [0, 1, 2]
    )
    volume = sum(n * q.get_vol_tot() for n, q in zip(weights, parts))
    area = sum(n * np.pi * q.get_Dwet() ** 2 for n, q in zip(weights, parts))
    assert r_eff(p) == pytest.approx(
        3 * volume / area, rel=1e-12, abs=0.0
    )
    assert r_eff(p) == pytest.approx(2.5e-6, rel=1e-12, abs=0.0)


def test_effective_radius_is_registered_with_build_variable():
    p = pop([2e-6, 6e-6], [3.0, 1.0])
    var = build_variable("effective_radius")
    assert var.meta.units == "m"
    assert var.compute(p, as_dict=True) == {"effective_radius": r_eff(p)}
