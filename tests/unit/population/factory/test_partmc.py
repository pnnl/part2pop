# tests/unit/population/factory/test_partmc.py

import numpy as np
import pytest
from pathlib import Path

# Try importing PARTMC factory & netCDF4; if unavailable, skip the tests.
try:
    import netCDF4  # noqa: F401
    from pyparticle.population.factory.partmc import build as build_partmc
    HAS_NETCDF4 = True
except Exception:
    HAS_NETCDF4 = False


pytestmark = pytest.mark.skipif(
    not HAS_NETCDF4,
    reason="netCDF4 not available; skipping PARTMC population factory tests.",
)


def _find_repo_root(start: Path) -> Path:
    """
    Walk up from `start` until we find the repo root containing the
    'examples_old' directory. This makes the test robust to where pytest
    is invoked from, as long as the directory layout is unchanged.
    """
    cur = start
    while cur != cur.parent:
        if (cur / "examples_old").is_dir():
            return cur
        cur = cur.parent
    raise RuntimeError("Could not locate repo root with 'examples_old' directory")


def test_partmc_build_real_data_rescales_to_requested_Ntot():
    """
    Use the real PARTMC example data shipped with the repository to test that:

      - The PARTMC factory successfully builds a population.
      - The number of particles equals the requested `n_particles`.
      - The total number concentration equals the requested `N_tot`.

    This exercises the *success path* of the factory, not just error handling.
    """
    # Locate repo root and PARTMC example directory
    this_file = Path(__file__).resolve()
    repo_root = _find_repo_root(this_file)
    partmc_dir = repo_root / "examples_old" / "example_data" / "model_output" / "partmc"

    assert partmc_dir.is_dir(), f"Expected PARTMC directory not found: {partmc_dir}"
    assert (partmc_dir / "out").is_dir(), "Expected 'out' subdirectory with NetCDF files"

    # We know from the example data that files like 05_0001_00000001.nc exist,
    # so using timestep=1 and repeat=1 is safe.
    cfg = {
        "partmc_dir": str(partmc_dir),
        "timestep": 1,
        "repeat": 1,
        "n_particles": 500,     # downsample to a manageable number
        "N_tot": 1.0e8,         # target total number concentration
        # leave species_modifications, specdata_path, add_mixing_ratios at defaults
    }

    pop = build_partmc(cfg)

    # 1) We built a non-empty population
    assert len(pop.ids) > 0

    # 2) The number of particles equals n_particles (sampling logic)
    assert len(pop.ids) == cfg["n_particles"]

    # 3) The total number concentration matches N_tot (within tight tolerance)
    N_tot = float(pop.get_Ntot())
    assert np.isclose(N_tot, cfg["N_tot"], rtol=1e-6)

    # 4) Basic sanity on masses: non-negative, finite, and at least one species
    assert pop.spec_masses.ndim == 2
    n_particles, n_species = pop.spec_masses.shape
    assert n_particles == cfg["n_particles"]
    assert n_species >= 1
    assert np.all(pop.spec_masses >= 0.0)
    assert np.all(np.isfinite(pop.spec_masses))

    # 5) Aerosol ids are integers and aligned with num_concs
    assert len(pop.ids) == pop.num_concs.size
    assert all(isinstance(i, int) for i in pop.ids)
    assert np.all(pop.num_concs > 0.0)

# tests/unit/population/factory/test_partmc_mass_composition.py

import numpy as np
import pytest
from pathlib import Path

# Try importing PARTMC factory & netCDF4; if unavailable, skip the tests.
try:
    import netCDF4  # noqa: F401
    from pyparticle.population.factory.partmc import build as build_partmc, get_ncfile
    HAS_NETCDF4 = True
except Exception:
    HAS_NETCDF4 = False


pytestmark = pytest.mark.skipif(
    not HAS_NETCDF4,
    reason="netCDF4 not available; skipping PARTMC mass-composition tests.",
)


def _find_repo_root(start: Path) -> Path:
    """
    Walk up from `start` until we find the repo root containing the
    'examples' directory. This makes the test robust to where pytest
    is invoked from (top-level, src/, etc.).
    """
    cur = start
    while cur != cur.parent:
        if (cur / "examples").is_dir():
            return cur
        cur = cur.parent
    raise RuntimeError("Could not locate repo root with 'examples' directory")


def test_partmc_mass_composition_matches_raw():
    """
    Use real PARTMC example data to verify that the constructed population
    preserves per-particle mass composition:

      - Select a subset of particles using the SAME random sampling used
        inside the factory (by controlling the NumPy RNG seed).
      - For each sampled particle id, compare:

          spec_masses_raw[:, ii]  (from NetCDF)
          vs
          spec_masses_pop[j, :]   (from ParticlePopulation, same species order)

        They should be identical (within tiny floating-point tolerance).
    """
    # ------------------------------------------------------------------
    # 1. Locate the PARTMC example directory used by the factory
    # ------------------------------------------------------------------
    this_file = Path(__file__).resolve()
    repo_root = _find_repo_root(this_file)

    partmc_dir = repo_root / "examples" / "example_data" / "model_output_old" / "partmc"
    out_dir = partmc_dir / "out"

    assert partmc_dir.is_dir(), f"Expected PARTMC directory not found: {partmc_dir}"
    assert out_dir.is_dir(), f"Expected PARTMC 'out' directory not found: {out_dir}"

    # We know the example data includes files like 05_0001_00000001.nc,
    # so using timestep=1, repeat=1 is safe.
    timestep = 1
    repeat = 1
    n_particles = 200  # subset to keep things quick

    # ------------------------------------------------------------------
    # 2. Open the same NetCDF file the factory will use and compute the
    #    sampled indices with a fixed RNG seed.
    # ------------------------------------------------------------------
    from netCDF4 import Dataset

    nc_path = get_ncfile(out_dir, timestep=timestep, repeat=repeat)
    assert nc_path.is_file(), f"Expected PARTMC NetCDF file not found: {nc_path}"

    ds = Dataset(nc_path, "r")

    # These are what the factory uses internally
    spec_masses_raw = np.array(ds.variables["aero_particle_mass"][:])  # (nspec, npart)
    part_ids_raw = np.array([pid for pid in ds.variables["aero_id"][:]], dtype=int)

    if "aero_num_conc" in ds.variables:
        num_concs_raw = np.array(ds.variables["aero_num_conc"][:], dtype=float)
    else:
        num_concs_raw = 1.0 / np.array(ds.variables["aero_comp_vol"][:], dtype=float)

    ds.close()

    n_raw = len(part_ids_raw)
    assert spec_masses_raw.shape[1] == n_raw
    assert num_concs_raw.shape[0] == n_raw

    assert n_particles < n_raw, "n_particles should be smaller than available raw particles"

    # Use the exact same RNG call as the factory:
    #   idx = np.random.choice(np.arange(len(part_ids)), size=n_particles, replace=False)
    seed = 12345
    np.random.seed(seed)
    idx = np.random.choice(np.arange(n_raw), size=n_particles, replace=False)

    sampled_ids_raw = part_ids_raw[idx]
    sampled_masses_raw = spec_masses_raw[:, idx]  # shape (nspec, n_particles)

    # ------------------------------------------------------------------
    # 3. Call the real PARTMC factory with the same RNG seed so it
    #    samples the same set of particle indices.
    # ------------------------------------------------------------------
    cfg = {
        "partmc_dir": str(partmc_dir),
        "timestep": timestep,
        "repeat": repeat,
        "n_particles": n_particles,
        "N_tot": 1.0e8,  # any positive number; rescaling of num_concs won't affect per-particle masses
        # species_modifications, specdata_path, add_mixing_ratios -> defaults
    }

    # Reset RNG so the factory sees the same sequence
    np.random.seed(seed)
    pop = build_partmc(cfg)

    # Sanity check: number of particles matches requested number
    assert len(pop.ids) == n_particles
    assert pop.spec_masses.shape[0] == n_particles

    # ------------------------------------------------------------------
    # 4. For each sampled raw particle id, locate the corresponding row
    #    in pop.spec_masses and compare the mass vectors.
    # ------------------------------------------------------------------
    # Build mapping from id -> row index in population
    id_to_idx_pop = {pid: j for j, pid in enumerate(pop.ids)}

    # For each sampled raw index/ID, compare spec_masses across all species
    for k, raw_id in enumerate(sampled_ids_raw):
        assert raw_id in id_to_idx_pop, f"Sampled particle id {raw_id} not found in population ids"

        j = id_to_idx_pop[raw_id]

        masses_raw = sampled_masses_raw[:, k]       # (nspec,)
        masses_pop = pop.spec_masses[j, :]          # (nspec,)

        # The factory does not alter per-particle mass composition; it only
        # rescales number concentrations. So these should match up to
        # floating-point tolerance.
        assert masses_pop.shape == masses_raw.shape
        assert np.allclose(masses_pop, masses_raw, rtol=0, atol=1e-14)

