# tests/unit/population/factory/test_partmc.py

import numpy as np
import pytest
from pathlib import Path

# Try importing PARTMC factory & netCDF4; if unavailable, skip the tests.
try:
    import netCDF4  # noqa: F401
    from part2pop.population.factory.partmc import (
        build as build_partmc,
        map_camp_specs,
        get_ncfile,
    )
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


def test_partmc_build_real_data_rescales_to_requested_Ntot(monkeypatch):
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
    nc_files = sorted(partmc_dir.glob("out/*.nc"))
    if not nc_files:
        pytest.skip(f"No PARTMC netCDF files found under: {partmc_dir}")
    
    infile = nc_files[0]

    # Load raw data for later comparison
    import netCDF4
    with netCDF4.Dataset(infile, "r") as ds:
        raw_masses = np.array(ds.variables["aero_particle_mass"][:])  # (nspec, npart)
        raw_ids = np.array(ds.variables["aero_id"][:], dtype=int)

    # Target total number concentration [m^-3] and requested number of particles
    N_tot_target = 1.0e8
    n_particles_target = min(200, raw_ids.size)

    # Deterministic selection of particles for reproducible assertions
    monkeypatch.setattr(
        "part2pop.population.factory.partmc.np.random.choice",
        lambda arr, size, replace: arr[:size],
    )

    pop = build_partmc(
        {
            "partmc_dir": str(partmc_dir),
            "timestep": 1,
            "repeat": 1,
            "N_tot": N_tot_target,
            "n_particles": n_particles_target,
            "suppress_warning": True,
        }
    )

    # Basic sanity checks on the constructed population
    assert len(pop.ids) == n_particles_target

    # Total number concentration should match the requested value (within FP noise)
    total_num = pop.get_Ntot() if hasattr(pop, "get_Ntot") else np.sum(pop.num_concs)
    assert np.isclose(total_num, N_tot_target, rtol=1e-12, atol=0.0)

    # The factory resamples particles but should not alter per-particle
    # mass composition. For a subset of particles we check that the
    # species mass vectors are consistent with the original data.
    sample_size = min(10, n_particles_target)
    sampled_indices = np.arange(sample_size)
    sampled_ids_raw = raw_ids[sampled_indices]
    nspec_pop = pop.spec_masses.shape[1]
    sampled_masses_raw = raw_masses[:nspec_pop, sampled_indices].T  # (nsample, nspec_pop)

    # Build a mapping from id -> index in the population for quick lookup
    id_to_idx_pop = {int(pid): i for i, pid in enumerate(pop.ids)}

    for k, raw_id in enumerate(sampled_ids_raw):
        assert raw_id in id_to_idx_pop, f"Sampled particle id {raw_id} not found in population ids"

        j = id_to_idx_pop[raw_id]

        # masses_raw: (nspec,), masses_pop: (nspec,)
        masses_raw = sampled_masses_raw[k, :]
        masses_pop = pop.spec_masses[j, :]

        assert masses_pop.shape == masses_raw.shape
        assert np.allclose(masses_pop, masses_raw, rtol=0, atol=1e-14)


def test_map_camp_specs_returns_suffixes():
    names = ["abc.def", "ghi"]
    assert map_camp_specs(names) == ["def", "ghi"]


def test_get_ncfile_requires_existing_directory(tmp_path):
    missing_dir = tmp_path / "out"
    with pytest.raises(FileNotFoundError):
        get_ncfile(missing_dir, timestep=1, repeat=1)


def test_get_ncfile_detects_known_prefix(tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True)
    file_path = out_dir / "urban_plume_0001_00000001.nc"
    file_path.write_text("data")
    result = get_ncfile(out_dir, timestep=1, repeat=1)
    assert result.name == "urban_plume_0001_00000001.nc"
