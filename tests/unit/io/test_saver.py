import importlib

import numpy as np

from part2pop.aerosol_particle import make_particle
from part2pop.io import loader, saver
from part2pop.population.base import ParticlePopulation


def test_import_io_saver():
    importlib.import_module("part2pop.io.saver")


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


def test_save_and_load_population_roundtrip(tmp_path):
    population = _simple_population()
    population.extra_attr = np.array([42])
    analysis_results = {
        "foo": {
            "config": {"value": np.int64(2)},
            "result": {"value": np.array([1.0, 2.0])},
        }
    }
    particle_results = {"bar": {"result": {"value": np.array([3.0])}}}
    metadata = {"tag": "roundtrip"}

    saved = saver.save_population(
        tmp_path / "data" / "pop.npz",
        population,
        analysis_results=analysis_results,
        particle_results=particle_results,
        metadata=metadata,
    )

    loaded_pop, analysis_records, loaded_metadata = loader.load_population(saved)
    assert np.allclose(loaded_pop.num_concs, population.num_concs)
    assert "foo" in analysis_records
    assert loaded_metadata["user_metadata"]["tag"] == "roundtrip"
    assert hasattr(loaded_pop, "extra_attr")
    assert np.array_equal(loaded_pop.extra_attr, population.extra_attr)

    loaded_pop, analysis_records, loaded_metadata, particle_records = loader.load_population(
        saved,
        include_particle_data=True,
    )
    assert "bar" in particle_records
