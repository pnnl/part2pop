"""The HiScale builder's annotations must resolve at runtime.

The module uses ``from __future__ import annotations``, so it imports without
``ParticlePopulation`` in scope; the missing name only surfaces when the
annotation is evaluated, for example by ``typing.get_type_hints`` (used by
documentation tools, type checkers and dataclass-style introspection).
"""

import typing

from part2pop.population.base import ParticlePopulation
from part2pop.population.factory import hiscale_observations


def test_build_return_annotation_resolves_to_particle_population():
    hints = typing.get_type_hints(hiscale_observations.build)
    assert hints["return"] is ParticlePopulation
