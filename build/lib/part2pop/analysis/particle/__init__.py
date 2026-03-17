"""Particle-scoped analysis primitives and discovery helpers."""
from .base import ParticleVariable
from .factory.registry import (
    register_particle_variable,
    get_particle_builder,
    list_particle_variables,
    describe_particle_variable,
)

__all__ = [
    "ParticleVariable",
    "register_particle_variable",
    "get_particle_builder",
    "list_particle_variables",
    "describe_particle_variable",
]
