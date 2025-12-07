
# viz/builder.py
from .factory.registry import discover_plotter_types

class PlotBuilder:
    """A class to build plotters from data."""
    def __init__(self, type: str, config: dict):
        self.type = type
        self.config = config
    
    def build(self):
        if not self.type:
            raise ValueError("PlotBuilder requires a 'type' to build a plotter.")
        types = discover_plotter_types()
        if self.type not in types:
            raise ValueError(f"Unknown plotter type: {self.type}")
        cls_or_factory = types[self.type]
        # Expect a class or callable that accepts (varname, var_cfg)
        return cls_or_factory(self.config)

def build_plotter(type, config):
    """Placeholder for future plotter builder logic."""
    return PlotBuilder(type, config).build()
