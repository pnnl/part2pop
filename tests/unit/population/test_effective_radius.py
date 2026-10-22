import numpy as np
import pytest

from part2pop import ParticlePopulation, make_particle


def pop(diameters, weights):
    parts = [make_particle(d, ["SO4"], [1.0]) for d in diameters]
    return ParticlePopulation(
        parts[0].species,
        np.array([p.masses for p in parts]),
        np.array(weights),
        list(range(len(parts))),
    )


def test_effective_radius_is_not_number_mean_radius():
    assert pop([2e-6, 6e-6], [1.0, 1.0]).get_effective_radius() == pytest.approx(
        2.8e-6, rel=1e-12, abs=0.0
    )


def test_unequal_concentrations_and_scale_invariance():
    p = pop([2e-6, 6e-6], [3.0, 1.0])
    assert p.get_effective_radius() == pytest.approx(2.5e-6, rel=1e-12, abs=0.0)
    p.num_concs *= 100.0
    assert p.get_effective_radius() == pytest.approx(2.5e-6, rel=1e-12, abs=0.0)
    assert pop([2e-6], [7.0]).get_effective_radius() == pytest.approx(
        1e-6, rel=1e-12, abs=0.0
    )


def test_zero_total_area_is_undefined():
    assert np.isnan(pop([2e-6], [0.0]).get_effective_radius())
    empty = ParticlePopulation((), np.empty((0, 0)), np.array([]), [])
    assert np.isnan(empty.get_effective_radius())


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
    assert p.get_effective_radius() == pytest.approx(
        3 * volume / area, rel=1e-12, abs=0.0
    )
    assert p.get_effective_radius() == pytest.approx(2.5e-6, rel=1e-12, abs=0.0)
