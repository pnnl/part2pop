import importlib

import numpy as np
import pytest

from part2pop.aerosol_particle import make_particle
from part2pop.io import loader, saver
from part2pop.population.base import ParticlePopulation


def test_import_io_loader():
    importlib.import_module("part2pop.io.loader")


def _simple_population():
    particle = make_particle(
        D=1e-6,
        aero_spec_names=["SO4"],
        aero_spec_frac=[1.0],
        species_modifications={},
        D_is_wet=True,
    )
    return ParticlePopulation(
        species=particle.species,
        spec_masses=np.atleast_2d(particle.masses).copy(),
        num_concs=np.array([1.0], dtype=float),
        ids=[1],
    )


def test_loader_wraps_scalar_results(tmp_path):
    population = _simple_population()
    analysis_results = {"foo": {"result": 7.0}}
    particle_results = {"bar": {"result": 3.0}}

    saved = saver.save_population(
        tmp_path / "data" / "pop.npz",
        population,
        analysis_results=analysis_results,
        particle_results=particle_results,
    )

    _, analysis_records, _, particle_records = loader.load_population(
        saved,
        include_particle_data=True,
    )

    assert "foo" in analysis_records
    assert list(analysis_records["foo"]["result"].keys()) == ["value"]
    assert analysis_records["foo"]["result"]["value"] == pytest.approx(7.0)
    assert "bar" in particle_records
    assert particle_records["bar"]["result"]["value"] == pytest.approx(3.0)
