# tests/unit/population/factory/test_monodisperse.py

import numpy as np
import pytest

from part2pop import build_population, ParticlePopulation


def test_monodisperse_single_particle_basic_properties():
    """
    Build a simple monodisperse population and check:
      - type is ParticlePopulation
      - we get exactly one particle
      - total number concentration matches the requested N
      - SO4 mass concentration is positive
    """
    cfg = {
        "type": "monodisperse",
        "aero_spec_names": [["SO4"]],
        "N": [2.0],
        "D": [1e-7],
        "aero_spec_fracs": [[1.0]],
    }

    pop = build_population(cfg)

    assert isinstance(pop, ParticlePopulation)
    assert len(pop.ids) == 1

    N_tot = pop.get_Ntot()
    assert np.isclose(N_tot, 2.0)

    m_so4 = pop.get_mass_conc("SO4")
    assert m_so4 > 0.0


def test_monodisperse_multiple_particles_number_conservation():
    """
    With multiple monodisperse entries, the population should:
      - contain one particle per entry in N/D
      - conserve total number concentration (sum of N)
    """
    cfg = {
        "type": "monodisperse",
        "aero_spec_names": [["SO4"]],
        "N": [1.0, 3.0, 5.0],
        "D": [0.05e-6, 0.10e-6, 0.20e-6],
        "aero_spec_fracs": [[1.0], [1.0], [1.0]],
    }

    pop = build_population(cfg)

    assert isinstance(pop, ParticlePopulation)
    assert len(pop.ids) == len(cfg["N"])

    N_expected = float(sum(cfg["N"]))
    N_tot = pop.get_Ntot()
    assert np.isclose(N_tot, N_expected)

    # All particles should have positive mass and number concentration
    assert np.all(pop.num_concs > 0.0)
    assert np.all(np.sum(pop.spec_masses, axis=1) > 0.0)


def test_monodisperse_accepts_stringy_numbers():
    cfg = {
        "type": "monodisperse",
        "aero_spec_names": [["SO4"]],
        "N": ["3.0"],
        "D": ["0.10e-6"],
        "aero_spec_fracs": [[1.0]],
    }

    pop = build_population(cfg)

    assert len(pop.ids) == 1
    assert np.isclose(pop.get_Ntot(), 3.0)


def test_monodisperse_resolves_alias_species_names_to_canonical_species_names():
    alias_cfg = {
        "type": "monodisperse",
        "aero_spec_names": [["dust", "soot", "org"]],
        "N": [2.0],
        "D": [1e-7],
        "aero_spec_fracs": [[0.2, 0.3, 0.5]],
    }
    canonical_cfg = {
        "type": "monodisperse",
        "aero_spec_names": [["OIN", "BC", "OC"]],
        "N": [2.0],
        "D": [1e-7],
        "aero_spec_fracs": [[0.2, 0.3, 0.5]],
    }

    alias_pop = build_population(alias_cfg)
    canonical_pop = build_population(canonical_cfg)

    alias_names = [sp.name for sp in alias_pop.species]
    canonical_names = [sp.name for sp in canonical_pop.species]
    assert alias_names == canonical_names
    assert alias_names[:3] == ["OIN", "BC", "OC"]
    assert np.isclose(alias_pop.get_Ntot(), canonical_pop.get_Ntot())


def test_monodisperse_aligns_fractions_with_per_particle_species_order():
    """
    Each particle's mass fractions must follow its own species names, even
    when the species are listed in a different order for each particle.
    """
    cfg = {
        "type": "monodisperse",
        "aero_spec_names": [["SO4", "OC"], ["OC", "SO4"]],
        "N": [1.0, 1.0],
        "D": [1e-7, 1e-7],
        "aero_spec_fracs": [[0.9, 0.1], [0.8, 0.2]],
    }

    pop = build_population(cfg)

    names = [sp.name for sp in pop.species]
    masses = pop.spec_masses
    for i, expected in enumerate((0.9, 0.2)):
        dry = masses[i, names.index("SO4")] + masses[i, names.index("OC")]
        assert masses[i, names.index("SO4")] / dry == pytest.approx(expected)


def test_monodisperse_shared_species_row_with_multiple_particles():
    cfg = {
        "type": "monodisperse",
        "aero_spec_names": [["SO4"]],
        "N": [1.0, 2.0],
        "D": [1e-7, 2e-7],
        "aero_spec_fracs": [[1.0], [1.0]],
    }

    pop = build_population(cfg)

    assert len(pop.ids) == 2


def test_monodisperse_species_row_length_mismatch_raises():
    cfg = {
        "type": "monodisperse",
        "aero_spec_names": [["SO4", "OC"]],
        "N": [1.0],
        "D": [1e-7],
        "aero_spec_fracs": [[1.0]],
    }

    with pytest.raises(ValueError, match="mismatched"):
        build_population(cfg)


def test_monodisperse_dry_diameter_with_water_first_and_missing_water_rows():
    """
    With D_is_wet=False the dry diameter is preserved and any requested water
    mass fraction is honored, wherever H2O appears in the species row or if it
    is not listed at all.
    """
    cfg = {
        "type": "monodisperse",
        "aero_spec_names": [["H2O", "SO4"], ["SO4"]],
        "N": [1.0, 1.0],
        "D": [100e-9, 100e-9],
        "aero_spec_fracs": [[0.5, 0.5], [1.0]],
        "D_is_wet": False,
    }

    pop = build_population(cfg)

    for ident, water_fraction in zip(pop.ids, [0.5, 0.0]):
        particle = pop.get_particle(ident)
        assert np.all(np.isfinite(particle.masses))
        assert particle.get_Ddry() == pytest.approx(100e-9, rel=1e-12, abs=0.0)
        water_idx = next(i for i, spec in enumerate(particle.species) if spec.name == "H2O")
        assert particle.masses[water_idx] / particle.masses.sum() == pytest.approx(water_fraction)
