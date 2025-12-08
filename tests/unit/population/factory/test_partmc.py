# tests/unit/population/factory/test_partmc.py

import numpy as np
import pytest
from pathlib import Path

# Try importing PARTMC factory & netCDF4; if unavailable, skip the tests.
try:
    import netCDF4  # noqa: F401
    from part2pop.population.factory.partmc import build as build_partmc

    HAS_NETCDF4 = True
except Exception:
    HAS_NETCDF4 = False


pytestmark = pytest.mark.skipif(
    not HAS_NETCDF4, reason="netCDF4 or PARTMC factory not available"
)


def _find_repo_root(start: Path) -> Path:
    """
    Walk upward from `start` until we find the repository root
    (identified by the presence of pytest.ini).
    """
    cur = start
    for _ in range(10):
        if (cur / "pytest.ini").is_file():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    # Fall back to the starting directory if we didn't find anything better
    return start


def test_partmc_build_real_data_rescales_to_requested_Ntot():
    """
    Use the real PARTMC example data (if present) to test that:

      - The PARTMC factory successfully builds a population.
      - The number of particles equals the requested `n_particles`.
      - The total number concentration equals the requested `N_tot`.
      - The per-particle mass composition is preserved.

    If the example data are not shipped with this checkout, the test
    is skipped rather than failing.
    """
    this_file = Path(__file__).resolve()
    repo_root = _find_repo_root(this_file)

    # This is the historical location used by the original tests.
    partmc_dir = repo_root / "examples" / "example_data" / "model_output" / "partmc"

    if not partmc_dir.is_dir():
        pytest.skip(f"PARTMC example data directory not found: {partmc_dir}")

    # Choose a single snapshot file; in the original repo this was
    # typically something like 'partmc_output.nc'.
    nc_files = sorted(partmc_dir.glob("*.nc"))
    if not nc_files:
        pytest.skip(f"No PARTMC netCDF files found under: {partmc_dir}")

    infile = nc_files[0]

    # Target total number concentration [m^-3] and requested number of particles
    N_tot_target = 1.0e8
    n_particles_target = 200

    pop = build_partmc(
        infile=str(infile),
        N_tot=N_tot_target,
        n_particles=n_particles_target,
    )

    # Basic sanity checks on the constructed population
    assert pop.n_particles == n_particles_target

    # Total number concentration should match the requested value (within FP noise)
    assert np.isclose(pop.N_tot, N_tot_target, rtol=1e-12, atol=0.0)

    # The factory resamples particles but should not alter per-particle
    # mass composition. For a subset of particles we check that the
    # species mass vectors are consistent with the original data.
    sampled_ids_raw = pop.ids[: min(10, pop.n_particles)]
    sampled_masses_raw = pop.spec_masses[: min(10, pop.n_particles), :].T  # (nspec, nsample)

    # Build a mapping from id -> index in the population for quick lookup
    id_to_idx_pop = {int(pid): i for i, pid in enumerate(pop.ids)}

    for k, raw_id in enumerate(sampled_ids_raw):
        assert raw_id in id_to_idx_pop, f"Sampled particle id {raw_id} not found in population ids"

        j = id_to_idx_pop[raw_id]

        # masses_raw: (nspec,), masses_pop: (nspec,)
        masses_raw = sampled_masses_raw[:, k]
        masses_pop = pop.spec_masses[j, :]

        assert masses_pop.shape == masses_raw.shape
        assert np.allclose(masses_pop, masses_raw, rtol=0, atol=1e-14)
