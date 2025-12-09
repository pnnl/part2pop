# tests/unit/population/test_base.py

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest
from types import SimpleNamespace

from part2pop.population.base import ParticlePopulation
from part2pop.aerosol_particle import make_particle
from part2pop.species.registry import get_species


def _empty_population():
    """Create a truly empty ParticlePopulation with the right shapes."""
    # Start with a dummy species entry; will be overwritten on first add_particle
    dummy_species = ()
    spec_masses = np.zeros((0, 0))
    num_concs = np.zeros((0,))
    ids = []
    return ParticlePopulation(
        species=dummy_species,
        spec_masses=spec_masses,
        num_concs=num_concs,
        ids=ids,
    )


def _make_simple_particle():
    sulfate = get_species("SO4")
    D_wet = 0.1e-6
    return make_particle(
        D=D_wet,
        aero_spec_names=[sulfate.name],
        aero_spec_frac=np.array([1.0]),
        species_modifications={},
        D_is_wet=True,
    )


def test_particle_population_add_and_get_particle():
    """
    Start from an empty ParticlePopulation and use add_particle via set_particle
    to populate it, then recover the particle and its number concentration.
    """
    p = _make_simple_particle()
    pop = _empty_population()

    # set_particle will call add_particle for new id
    pop.set_particle(particle=p, part_id=5, num_conc=1e8, suppress_warning=False)

    assert 5 in pop.ids
    idx = list(pop.ids).index(5)
    assert np.isclose(pop.num_concs[idx], 1e8)

    p_out = pop.get_particle(5)
    assert np.isclose(p_out.get_Dwet(), p.get_Dwet())
    assert np.isclose(p_out.get_mass_tot(), p.get_mass_tot())


def test_particle_population_set_particle_overwrites_existing():
    """
    When we call set_particle with an existing id, the masses and num_conc
    should be overwritten in place (not duplicated).
    """
    p1 = _make_simple_particle()
    pop = _empty_population()

    pop.set_particle(particle=p1, part_id=1, num_conc=1e8)
    assert len(pop.ids) == 1

    p2 = _make_simple_particle()
    # Change the number concentration
    pop.set_particle(particle=p2, part_id=1, num_conc=3e8)

    assert len(pop.ids) == 1  # still one entry
    idx = list(pop.ids).index(1)
    assert np.isclose(pop.num_concs[idx], 3e8)


def test_particle_population_total_number_concentration():
    """
    get_Ntot should sum num_concs over all stored particles.
    """
    p = _make_simple_particle()
    pop = _empty_population()

    pop.set_particle(particle=p, part_id=1, num_conc=1e8)
    pop.set_particle(particle=p, part_id=2, num_conc=2e8)

    N_tot = pop.get_Ntot()
    assert np.isclose(N_tot, 3e8)


# ---------------------------------------------------------------------------
# Extended tests using a stubbed Particle to isolate population logic
# ---------------------------------------------------------------------------


class _FakeParticle:
    def __init__(self, species, masses):
        self.species = species
        self.masses = np.array(masses, dtype=float)

    def get_Dwet(self, *args, **kwargs):
        # diameter in meters; here just proportional to total mass for simplicity
        return float(self.masses.sum())

    def idx_h2o(self):
        for i, spec in enumerate(self.species):
            if spec.name == "H2O":
                return i
        return -1

    def get_variable(self, varname, *args, **kwargs):
        # return sum of masses for any variable name
        return float(self.masses.sum())


def _make_stub_population(monkeypatch):
    # Two species including H2O
    species = (SimpleNamespace(name="H2O"), SimpleNamespace(name="SO4"))
    spec_masses = np.array([[1.0, 2.0]])
    num_concs = np.array([5.0])
    ids = [1]
    monkeypatch.setattr("part2pop.population.base.Particle", _FakeParticle)
    return ParticlePopulation(species, spec_masses, num_concs, ids)


def test_find_and_get_particle_stub(monkeypatch):
    pop = _make_stub_population(monkeypatch)
    assert pop.find_particle(1) == 0
    assert pop.find_particle(99) == len(pop.ids)

    particle = pop.get_particle(1)
    assert isinstance(particle, _FakeParticle)
    with pytest.raises(ValueError):
        pop.get_particle(99)


def test_set_and_add_particle_stub(monkeypatch):
    pop = _make_stub_population(monkeypatch)
    new_particle = _FakeParticle(pop.species, [3.0, 4.0])
    pop.set_particle(new_particle, part_id=1, num_conc=2.5)
    assert pop.num_concs[0] == 2.5
    assert pop.spec_masses[0, 0] == 3.0

    # Add a brand new particle id
    another = _FakeParticle(pop.species, [1.0, 1.0])
    pop.set_particle(another, part_id=2, num_conc=1.0)
    assert len(pop.ids) == 2
    assert pop.num_concs[-1] == 1.0


def test_population_mass_and_radius_stub(monkeypatch):
    pop = _make_stub_population(monkeypatch)
    # effective radius uses get_Dwet / 2 averaged by num_concs
    eff_r = pop.get_effective_radius()
    expected = (pop.get_particle(1).get_Dwet() / 2.0)
    assert eff_r == expected

    assert pop.get_tot_mass() == np.sum(pop.num_concs * np.sum(pop.spec_masses, axis=1))
    assert pop.get_tot_dry_mass() == pop.get_tot_mass() - pop.num_concs[0] * pop.spec_masses[0, 0]
    assert pop.get_mass_conc("SO4") == pop.num_concs[0] * pop.spec_masses[0, 1]


def test_get_particle_var_and_hist_stub(monkeypatch):
    pop = _make_stub_population(monkeypatch)
    vals = pop.get_particle_var("any")
    assert vals.shape == (len(pop.ids),)

    hist, edges = pop.get_num_dist_1d(N_bins=2, density=False)[0:2]
    assert hist.shape == (2,)

    with pytest.raises(NotImplementedError):
        pop.get_num_dist_1d(method="kde")


def _make_bc_population():
    bc = get_species("BC")
    so4 = get_species("SO4")
    h2o = get_species("H2O")
    species = (bc, so4, h2o)
    spec_masses = np.array([
        [1.0, 2.0, 0.5],  # BC, SO4, H2O
        [0.5, 1.0, 1.5],
    ])
    num_concs = np.array([1.0, 2.0])
    ids = [1, 2]
    pop = ParticlePopulation(species, spec_masses.copy(), num_concs.copy(), ids.copy())
    pop.num_conc = pop.num_concs  # fix typo used by reduce_mixing_state
    return pop


def test_reduce_mixing_state_same_dry_mass_and_same_bc():
    pop = _make_bc_population()
    with pytest.raises(Exception):
        pop.reduce_mixing_state("MAM5 sameDryMass", RH=0.5, T=298.15)
    with pytest.raises(Exception):
        pop.reduce_mixing_state("MAM5 sameBC", RH=0.5, T=298.15)


def test_reduce_mixing_state_part_res_raises_unindexed():
    pop = _make_bc_population()
    with pytest.raises(TypeError):
        pop.reduce_mixing_state("part_res", RH=0.4, T=280.0)


def _fake_population_result():
    species = (get_species("SO4"), get_species("OC"), get_species("H2O"))
    spec_masses = np.zeros((1, len(species)))
    num_concs = np.array([1.0])
    return SimpleNamespace(
        species=species,
        spec_masses=spec_masses,
        num_concs=num_concs,
        ids=[0],
    )


class _FakeDataset:
    def __init__(self, arrays, has_gsd=True):
        self._arrays = arrays
        self.variables = {"gsd_a": True} if has_gsd else {}

    def __getitem__(self, key):
        return self._arrays[key]

    def close(self):
        pass


def _reload_mam4_module(monkeypatch, netcdf_module):
    importlib.invalidate_caches()
    monkeypatch.setitem(sys.modules, "netCDF4", netcdf_module)
    if "part2pop.population.factory.mam4" in sys.modules:
        del sys.modules["part2pop.population.factory.mam4"]
    importlib.invalidate_caches()
    import part2pop.population.factory.mam4 as mam4_mod
    importlib.reload(mam4_mod)
    return mam4_mod


def _monkeypatch_build_binned(monkeypatch):
    captured = {}

    def fake_build(cfg):
        captured["cfg"] = cfg
        return _fake_population_result()

    monkeypatch.setattr("part2pop.population.factory.mam4.build_binned_lognormals", fake_build)
    return captured


def test_mam4_build_combines_namelist_and_stubs(monkeypatch, tmp_path):
    dummy_nc = SimpleNamespace(Dataset=lambda *args, **kwargs: None)
    mam4_mod = _reload_mam4_module(monkeypatch, dummy_nc)

    captured = _monkeypatch_build_binned(monkeypatch)
    namelist = tmp_path / "namelist"
    lines = []
    for idx in range(4):
        lines.append(f"numc{idx+1} = {1.0 + idx},")
        lines.append(f"mfso4{idx+1} = 0.6,")
        lines.append(f"mfpom{idx+1} = 0.4,")
    namelist.write_text("\n".join(lines))

    cfg = {
        "mam4_dir": str(tmp_path),
        "timestep": 1,
        "GSD": [1.6, 1.7, 1.8, 1.9],
        "N_bins": [3, 3, 3, 3],
    }
    pop = mam4_mod.build(cfg)
    assert isinstance(pop, SimpleNamespace)
    assert captured["cfg"]["type"] == "binned_lognormals"
    assert captured["cfg"]["N"].shape[0] == 4


def test_mam4_build_uses_output_nc_for_timestep_two(monkeypatch, tmp_path):
    arrays = {
        "num_aer": np.ones((4, 2)) * np.arange(1, 5).reshape(4, 1),
        "so4_aer": np.ones((4, 2)),
        "soa_aer": np.ones((4, 2)) * 2.0,
        "dgn_a": np.ones((4, 2)) * 1.2,
        "dgn_awet": np.ones((4, 2)) * 1.5,
    }
    fake_ds = _FakeDataset(arrays)
    dummy_nc = SimpleNamespace(Dataset=lambda *args, **kwargs: fake_ds)
    mam4_mod = _reload_mam4_module(monkeypatch, dummy_nc)
    captured = _monkeypatch_build_binned(monkeypatch)

    # create a dummy netCDF file so shutil.copy has something to grab
    (tmp_path / "mam_output.nc").write_text("dummy")

    cfg = {
        "mam4_dir": str(tmp_path),
        "timestep": 2,
        "GSD": [1.6, 1.6, 1.6, 1.6],
        "N_bins": [2, 2, 2, 2],
        "p": 101325,
        "T": 298.15,
    }
    pop = mam4_mod.build(cfg)
    assert isinstance(pop, SimpleNamespace)
    assert captured["cfg"]["type"] == "binned_lognormals"
    assert "D_is_wet" in captured["cfg"]


def test_reduce_mixing_state_part_res_raises_type_error():
    pop = _make_bc_population()
    with pytest.raises(TypeError):
        pop.reduce_mixing_state("part_res")
