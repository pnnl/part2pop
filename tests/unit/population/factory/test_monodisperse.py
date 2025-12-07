# tests/unit/population/factory/test_monodisperse.py

import numpy as np

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
