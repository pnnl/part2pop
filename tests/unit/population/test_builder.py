# tests/unit/population/test_builder.py

import importlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from part2pop import ParticlePopulation
from part2pop.population.builder import PopulationBuilder, build_population



def test_population_builder_missing_type_raises():
    """
    Configs without a 'type' key should raise a clear ValueError.
    """
    cfg = {}

    builder = PopulationBuilder(cfg)
    with pytest.raises(ValueError, match="type"):
        builder.build()

    # Convenience wrapper should behave the same
    with pytest.raises(ValueError, match="type"):
        build_population(cfg)


def test_population_builder_unknown_type_raises():
    """
    An unknown population type should raise a ValueError listing the type.
    """
    cfg = {
        "type": "this_type_does_not_exist",
    }

    with pytest.raises(ValueError, match="Unknown population type"):
        build_population(cfg)


def test_population_builder_dispatches_to_monodisperse_factory():
    """
    Given a monodisperse config, PopulationBuilder should use the
    factory registry to create a ParticlePopulation instance.
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
    # Sanity: total N should match requested
    assert pop.get_Ntot() == pytest.approx(2.0)


class _FakeMamDataset:
    def __init__(self):
        shape = (3, 2)
        self.data = {
            "num_aer": np.full(shape, 1e3),
            "so4_aer": np.full(shape, 2e3),
            "soa_aer": np.full(shape, 1e3),
            "dgn_a": np.full(shape, 1e-7),
            "dgn_awet": np.full(shape, 2e-7),
        }

    def __getitem__(self, key):
        return self.data[key]


def _reload_mam4_with_fake_nc(monkeypatch):
    fake_nc = types.SimpleNamespace(Dataset=lambda path: _FakeMamDataset())
    monkeypatch.setitem(sys.modules, "netCDF4", fake_nc)
    sys.modules.pop("part2pop.population.factory.mam4", None)
    return importlib.import_module("part2pop.population.factory.mam4")


def test_mam4_build_uses_fake_dataset(monkeypatch):
    mam4_mod = _reload_mam4_with_fake_nc(monkeypatch)

    captured = {}

    def fake_build(cfg):
        captured["cfg"] = cfg
        return "built"

    monkeypatch.setattr(mam4_mod, "build_binned_lognormals", fake_build)

    cfg = {
        "mam4_dir": Path("dummy"),
        "timestep": 2,
        "GSD": [1.5, 1.6, 1.7],
        "N_bins": [6, 6, 6],
        "p": 1e5,
        "T": 300.0,
        "D_is_wet": True,
    }

    result = mam4_mod.build(cfg)
    assert result == "built"
    assert captured["cfg"]["D_is_wet"]

    cfg["D_is_wet"] = False
    result = mam4_mod.build(cfg)
    assert captured["cfg"]["D_is_wet"] is False
