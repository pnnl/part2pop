import numpy as np
import pytest
from part2pop.analysis.builder import build_variable
from part2pop.analysis.population.factory.INSA_distribution import InsaDistVar, build

from part2pop import ParticlePopulation, make_particle


@pytest.mark.parametrize("through_builder", [False, True])
@pytest.mark.parametrize("first_normalized", [False, True])
def test_raw_and_normalized_metadata_are_independent(through_builder, first_normalized):
    def make(normalize):
        cfg = {"normalize": normalize}
        return (
            build_variable("INSA_distribution", var_cfg=cfg)
            if through_builder
            else build(cfg)
        )

    first = make(first_normalized)
    before = (first.meta.units, first.meta.long_label, first.meta.short_label)
    second = make(not first_normalized)
    assert first.meta is not second.meta
    assert first.meta is not InsaDistVar.meta
    assert (first.meta.units, first.meta.long_label, first.meta.short_label) == before
    raw, normalized = (second, first) if first_normalized else (first, second)
    assert raw.meta.units == "m$^{-3}$"
    assert raw.meta.long_label == "dN/dlnINSA"
    assert normalized.meta.units == ""
    assert normalized.meta.long_label == "Normalized dN/dlnINSA"
    assert InsaDistVar.meta.units == "m$^{-3}$"


def test_constructing_normalized_variable_preserves_raw_values_and_units():
    particle = make_particle(100e-9, ["OC", "H2O"], [1.0, 0.0])
    pop = ParticlePopulation(
        particle.species, np.array([particle.masses]), np.array([100.0]), [0]
    )
    edges = np.geomspace(1e-16, 1e-11, 50)
    raw = build({"method": "hist", "edges": edges})
    before = raw.compute(pop)
    build({"normalize": True})
    after = raw.compute(pop)
    np.testing.assert_array_equal(after, before)
    assert np.sum(after * np.diff(np.log(edges))) == pytest.approx(100.0)
    assert raw.meta.units == "m$^{-3}$"
