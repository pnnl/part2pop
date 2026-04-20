"""
Optics package public API.

Expose a minimal set of builder functions and types at package level so
other modules can import them via `from part2pop.optics import ...`.
"""

from .builder import build_freezing_particle, build_freezing_population
from .factory.utils import calculate_Psat, calculate_dPsat_dT

__all__ = ["build_freezing_particle", "build_freezing_population","calculate_Psat","calculate_dPsat_dT"]

