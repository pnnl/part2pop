from __future__ import annotations
from typing import Dict, Any
from ..base import Variable, VariableMeta

class ParticleVariable(Variable):
    """Base class for variables that operate on a single particle.

    Implementations should provide `meta: VariableMeta` and a `compute(particle)`
    method that returns a dict mapping axis names and the value keyed by
    `meta.name`.
    """
    meta: VariableMeta

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

    def compute(self, population):  # pragma: no cover - interface
        raise NotImplementedError

