from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Sequence, Tuple, Optional


#@dataclass(frozen=True)
@dataclass
class VariableMeta:
    name: str
    axis_names: Sequence[str]
    description: str  # axes label without units
    long_label: Optional[str] = None  # with units
    short_label: Optional[str] = None  # with units, for plots
    default_cfg: Dict[str, Any] = field(default_factory=dict)
    aliases: Tuple[str, ...] = ()
    units: Optional[str] = None
    scale: str = "linear"  # or 'log'
    
class Variable:
    """Base class for variables that operate on an entire population.

    This class replaces the legacy `AbstractVariable`. For backwards
    compatibility the top-level `analysis.base` module will alias
    `AbstractVariable = Variable`.
    """
    meta: VariableMeta
    def __init__(self, cfg: Dict[str, Any]):
        # VariableBuilder is responsible for merging defaults; Variable
        # instances simply accept the already-merged config.
        self.cfg = cfg
    
    def compute(self, population):  # pragma: no cover - interface
        raise NotImplementedError
    
    def rescale(self, new_units):
        """Rescale the variable to new units.

        This is a no-op if the variable has no units or if the new units
        are the same as the current units.
        """
        if self.meta.units is None or new_units == self.meta.units:
            pass
        raise NotImplementedError("Rescaling not implemented for " + new_units)