from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Sequence, Tuple
from ..base import Variable, VariableMeta


# fixme: call this something else? StateVariable? PopulationStateVariable?
class PopulationVariable(Variable):
    """Base class for variables that operate on an entire population.

    This class replaces the legacy `AbstractVariable`. For backwards
    compatibility the top-level `analysis.base` module will alias
    `AbstractVariable = PopulationVariable`.
    """
    meta: VariableMeta
    
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

    def compute(self, population):
        raise NotImplementedError

