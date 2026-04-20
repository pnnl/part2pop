"""
Optics package public API.

Expose a minimal set of builder functions and types at package level so
other modules can import them via `from part2pop.optics import ...`.
"""

from .builder import build_freezing_particle, build_freezing_population

__all__ = ["build_freezing_particle", "build_freezing_population"]

