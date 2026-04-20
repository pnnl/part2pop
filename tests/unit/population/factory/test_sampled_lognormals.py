# tests/unit/population/factory/test_sampled_lognormals.py

import numpy as np

from part2pop import build_population


def test_sampled_lognormals_population_size_and_total_number():
    """
    Build a sampled_lognormals population and ensure that:
      - The number of particle ids is <= N_parts.
      - The total number concentration is close to the target N.
    """
    cfg = {
        "type": "sampled_lognormals",
        "GMD": [0.1e-6],
        "GSD": [1.6],
        "N": [1e8],
        "N_parts": 5000,
        "aero_spec_names": [["SO4"]],
        "aero_spec_fracs": [[1.0]],
    }

    pop = build_population(cfg)

    # Number of particles in the population should be N_parts
    assert len(pop.ids) == cfg["N_parts"]

    N_tot = float(pop.get_Ntot())

    # Sampling introduces some noise, so allow looser tolerance
    assert np.isclose(N_tot, 1e8, rtol=0.1)


def test_sampled_lognormals_two_modes_reasonable_Ntot():
    """
    For two lognormal modes sampled into a single population, the total
    N should be close to the sum of both modes.
    """
    cfg = {
        "type": "sampled_lognormals",
        "GMD": [0.05e-6, 0.15e-6],
        "GSD": [1.4, 1.7],
        "N": [5e7, 2e8],
        "N_parts": 8000,
        "aero_spec_names": [["SO4"],["SO4"]],
        "aero_spec_fracs": [[1.0],[1.0]],
    }

    pop = build_population(cfg)

    assert len(pop.ids) > 0

    # Number of particles in the population should be N_parts
    print(len(pop.ids),cfg["N_parts"])
    assert len(pop.ids) == cfg["N_parts"]

    # Sampling introduces some noise, so allow looser tolerance
    N_requested = float(np.sum(cfg["N"]))
    N_tot = float(pop.get_Ntot())
    assert np.isclose(N_tot, N_requested, rtol=0.1)


def test_sampled_lognormals_accepts_stringy_numbers():
    n_parts_str = "3500"
    cfg = {
        "type": "sampled_lognormals",
        "GMD": ["0.1e-6"],
        "GSD": ["1.6"],
        "N": ["1e8"],
        "N_parts": n_parts_str,
        "aero_spec_names": [["SO4"]],
        "aero_spec_fracs": [[1.0]],
    }

    pop = build_population(cfg)

    assert len(pop.ids) == int(n_parts_str)
