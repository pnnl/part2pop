# tests/unit/analysis/test_builder.py

import pytest

from pyparticle.analysis.builder import VariableBuilder, build_variable
from pyparticle.analysis.population.factory.registry import UnknownVariableError
from pyparticle.analysis.particle.factory.registry import UnknownParticleVariableError


def test_variable_builder_unknown_population_var_raises():
    with pytest.raises(UnknownVariableError):
        VariableBuilder("not_a_real_var", {}, scope="population").build()


def test_variable_builder_unknown_particle_var_raises():
    with pytest.raises(UnknownParticleVariableError):
        VariableBuilder("not_a_real_var", {}, scope="particle").build()


def test_build_variable_population_dNdlnD():
    var = build_variable("dNdlnD", scope="population", var_cfg={})
    # Must have meta and a compute method
    assert hasattr(var, "meta")
    assert callable(var.compute)


def test_build_variable_particle_Dwet():
    var = build_variable("Dwet", scope="particle", var_cfg={})
    assert hasattr(var, "meta")
    assert callable(var.compute)
