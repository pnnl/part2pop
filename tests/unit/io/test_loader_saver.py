import numpy as np
import pytest

from part2pop.aerosol_particle import make_particle
from part2pop.io import loader, saver, _utils
from part2pop.population.base import ParticlePopulation


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


def test_make_json_safe_handles_nested_values():
    payload = {
        "array": np.array([1, 2]),
        "mapping": {"pi": np.float64(3.14)},
        "sequence": (np.int32(4), "text"),
    }
    safe = _utils.make_json_safe(payload)
    assert isinstance(safe["array"], list)
    assert safe["mapping"]["pi"] == 3.14
    assert safe["sequence"][0] == 4


def test_ensure_path_creates_parent_dirs(tmp_path):
    target = tmp_path / "nested" / "data" / "population"
    result = _utils.ensure_path(target)
    assert result == target
    assert result.parent.exists()


def test_save_and_load_population_roundtrip(tmp_path):
    population = _simple_population()
    population.extra_attr = np.array([42])

    analysis_results = {
        "foo": {
            "config": {"value": np.int64(2)},
            "result": {"value": np.array([1.0, 2.0])},
        }
    }
    particle_results = {
        "bar": {
            "result": {"value": np.array([3.0])},
        }
    }

    metadata = {"tag": "roundtrip"}
    out_file = tmp_path / "data" / "pop.npz"

    saved = saver.save_population(
        out_file,
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

    (
        loaded_pop,
        analysis_records,
        loaded_metadata,
        particle_records,
    ) = loader.load_population(saved, include_particle_data=True)
    assert "bar" in particle_records


def test_make_json_safe_handles_custom_objects():
    class Custom:
        def __str__(self):
            return "custom"

    payload = {
        "array": np.array([1, 2]),
        "nested": {"value": np.float64(3.0)},
        "tuple": (np.int32(4), "text"),
        "custom": Custom(),
    }
    safe = _utils.make_json_safe(payload)

    assert isinstance(safe["array"], list)
    assert safe["nested"]["value"] == 3.0
    assert safe["tuple"][0] == 4
    assert safe["custom"] == "custom"


def test_serialize_metadata_emits_numpy_string():
    metadata = {"tag": "roundtrip", "value": 42}
    serialized = _utils.serialize_metadata(metadata)

    assert isinstance(serialized, np.ndarray)
    assert serialized.dtype.type == np.str_
    assert "tag\":\"roundtrip" in serialized.item()


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
        saved, include_particle_data=True
    )

    assert "foo" in analysis_records
    assert list(analysis_records["foo"]["result"].keys()) == ["value"]
    assert analysis_records["foo"]["result"]["value"] == pytest.approx(7.0)

    assert "bar" in particle_records
    assert particle_records["bar"]["result"]["value"] == pytest.approx(3.0)
