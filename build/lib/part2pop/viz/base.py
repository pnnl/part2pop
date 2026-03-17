# viz/base.py
from dataclasses import dataclass

@dataclass
class Plotter:
    """A base class for plotters."""
    type: str
    config: dict

    def prep(self, population):
        raise NotImplementedError
            
    def plot(self, population, ax, **kwargs):
        raise NotImplementedError
