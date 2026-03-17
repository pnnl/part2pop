"""
Optics package public API.

Expose a minimal set of builder functions and types at package level so
other modules can import them via `from part2pop.optics import ...`.
"""

from .builder import build_optical_particle, build_optical_population
from .base import OpticalPopulation

__all__ = ["build_optical_particle", "build_optical_population", "OpticalPopulation"]

