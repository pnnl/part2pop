import numpy as np
import pytest

from part2pop import Particle, get_species


@pytest.fixture
def particle():
    return Particle(
        tuple(get_species(n, None) for n in ["Na", "NaCl", "H2O"]),
        [1e-18, 2e-18, 3e-18],
    )


def test_salt_lookup_excludes_sodium(particle):
    np.testing.assert_array_equal(particle.get_spec_mass("NaCl"), [2e-18])


def test_salt_update_preserves_sodium(particle):
    particle.set_spec_mass("NaCl", 7e-18)
    np.testing.assert_array_equal(particle.masses, [1e-18, 7e-18, 3e-18])


def test_exact_volume_moles_density_and_collection(particle):
    assert particle.get_spec_vol("NaCl").shape == (1,)
    assert particle.get_spec_moles("NaCl") == pytest.approx(
        2e-18 / 0.0585, rel=1e-12, abs=0.0
    )
    np.testing.assert_array_equal(particle.get_spec_rho("NaCl"), [2200.0])
    np.testing.assert_array_equal(particle.idx_spec(["Na", "H2O"])[0], [0, 2])
    assert particle.get_spec_mass("nacl").size == 0


@pytest.mark.parametrize("names", [("Na", "NaCl", "H2O"), ("H2O", "NaCl", "Na")])
def test_collection_and_absent_names_preserve_particle_order(names):
    p = Particle(tuple(get_species(n, None) for n in names), [1e-18, 2e-18, 3e-18])
    indices = [i for i, n in enumerate(names) if n in {"NaCl", "H2O"}]
    np.testing.assert_array_equal(p.idx_spec(("H2O", "NaCl"))[0], indices)
    np.testing.assert_array_equal(p.get_spec_mass(("H2O", "NaCl")), p.masses[indices])
    assert p.get_spec_mass("absent").size == 0
    assert p.get_spec_mass("aCl").size == 0


def test_generator_selector_matches_every_requested_name(particle):
    # A generator used to be consumed while testing the first species.
    selected = particle.idx_spec(name for name in ("H2O", "NaCl"))[0]
    np.testing.assert_array_equal(selected, [1, 2])
    np.testing.assert_array_equal(
        particle.get_spec_mass(name for name in ("Na", "H2O")), [1e-18, 3e-18]
    )
