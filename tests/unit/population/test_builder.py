# tests/unit/population/test_builder.py

import pytest

from pyparticle import ParticlePopulation
from pyparticle.population.builder import PopulationBuilder, build_population


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
