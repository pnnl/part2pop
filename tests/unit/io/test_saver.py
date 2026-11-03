import importlib

import numpy as np
import pytest

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
    population.metadata = {
        "source": "roundtrip",
        "nested": {"values": np.array([1, 2], dtype=np.int32)},
    }
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

    with np.load(saved, allow_pickle=False) as archive:
        assert "extra__metadata" not in archive.files
        assert any(key.startswith("population_metadata__array_") for key in archive.files)

    loaded_pop, analysis_records, loaded_metadata = loader.load_population(saved)
    assert np.allclose(loaded_pop.num_concs, population.num_concs)
    assert "foo" in analysis_records
    assert loaded_metadata["user_metadata"]["tag"] == "roundtrip"
    assert hasattr(loaded_pop, "extra_attr")
    assert np.array_equal(loaded_pop.extra_attr, population.extra_attr)
    assert loaded_pop.metadata["source"] == "roundtrip"
    np.testing.assert_array_equal(loaded_pop.metadata["nested"]["values"], np.array([1, 2], dtype=np.int32))
    assert loaded_pop.metadata["nested"]["values"].dtype == np.dtype(np.int32)
    np.testing.assert_array_equal(
        loaded_metadata["population_metadata"]["nested"]["values"],
        np.array([1, 2], dtype=np.int32),
    )

    loaded_pop, analysis_records, loaded_metadata, particle_records = loader.load_population(
        saved,
        include_particle_data=True,
    )
    assert "bar" in particle_records


def test_population_metadata_roundtrip_preserves_hiscale_shaped_arrays(tmp_path):
    population = _simple_population()
    population.metadata = {
        "source": "hiscale_observations",
        "size_distribution": {
            "Dp_lo_nm": np.array([100.0, 200.0], dtype=np.float32),
            "Dp_hi_nm": np.array([200.0, 400.0], dtype=np.float64),
            "N_bin_m3": np.array([10, 20], dtype=np.int64),
        },
    }

    saved = saver.save_population(tmp_path / "hiscale_metadata.npz", population)
    loaded_pop, _, _ = loader.load_population(saved)

    loaded = loaded_pop.metadata["size_distribution"]
    for name, original in population.metadata["size_distribution"].items():
        assert isinstance(loaded[name], np.ndarray)
        assert loaded[name].shape == original.shape
        assert loaded[name].dtype == original.dtype
        np.testing.assert_array_equal(loaded[name], original)


def test_save_and_load_population_without_metadata_preserves_absence(tmp_path):
    population = _simple_population()

    saved = saver.save_population(tmp_path / "no_metadata.npz", population)
    loaded_pop, _, loaded_metadata = loader.load_population(saved)

    assert not hasattr(loaded_pop, "metadata")
    assert "population_metadata" not in loaded_metadata


def test_population_metadata_object_array_is_rejected(tmp_path):
    population = _simple_population()
    population.metadata = {"unsupported": np.array([{"a": 1}], dtype=object)}

    with pytest.raises(TypeError, match="object dtype"):
        saver.save_population(tmp_path / "object_metadata.npz", population)


@pytest.mark.parametrize("dtype", [np.longdouble, np.clongdouble])
def test_population_metadata_extended_scalar_roundtrip(tmp_path, dtype):
    # Use a value with more precision than a Python float can retain where
    # the platform provides extended precision. Include an imaginary part too.
    value = dtype("1.234567890123456789")
    if np.issubdtype(dtype, np.complexfloating):
        value += dtype(1j) * dtype("0.1234567890123456789")
    if not isinstance(value.item(), np.generic):
        pytest.skip("scalar has a native Python representation on this platform")
    population = _simple_population()
    population.metadata = {"nested": {"value": value}}
    saved = saver.save_population(tmp_path / "extended_scalar.npz", population)
    with np.load(saved, allow_pickle=False) as archive:
        assert all(not archive[key].dtype.hasobject for key in archive.files)
    restored, _, _ = loader.load_population(saved)
    result = restored.metadata["nested"]["value"]
    assert isinstance(result, np.generic)
    assert result.dtype == value.dtype
    assert result == value


def test_population_metadata_masked_array_is_rejected(tmp_path):
    # np.asarray() would drop the mask and restore masked entries as data.
    population = _simple_population()
    population.metadata = {"flags": np.ma.masked_array([1.0, 2.0], mask=[False, True])}

    with pytest.raises(TypeError, match="masked arrays"):
        saver.save_population(tmp_path / "masked_metadata.npz", population)


def test_loaded_metadata_is_independent_of_population_metadata(tmp_path):
    population = _simple_population()
    population.metadata = {"source": "roundtrip", "nested": {"values": np.array([1, 2])}}
    saved = saver.save_population(tmp_path / "independent.npz", population)

    loaded_pop, _, loaded_metadata = loader.load_population(saved)
    loaded_pop.metadata["source"] = "edited"
    loaded_pop.metadata["nested"]["values"][0] = 99

    assert loaded_metadata["population_metadata"]["source"] == "roundtrip"
    np.testing.assert_array_equal(
        loaded_metadata["population_metadata"]["nested"]["values"], [1, 2]
    )
