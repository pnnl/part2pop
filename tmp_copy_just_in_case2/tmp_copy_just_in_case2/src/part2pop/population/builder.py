# src/part2pop/population/builder.py

"""Population builder wrapper used to construct particle populations from configs.

Supports discovery of population types via a registry and instantiates the
appropriate population class using the provided configuration dictionary.
"""

from .factory.registry import discover_population_types


class PopulationBuilder:
    """Construct a ParticlePopulation instance from a configuration dict."""
    def __init__(self, config):
        self.config = config
    
    def build(self):
        type_name = self.config.get("type")
        if not type_name:
            raise ValueError("Config must include a 'type' key.")
        types = discover_population_types()
        if type_name not in types:
            raise ValueError(f"Unknown population type: {type_name}")
        cls = types[type_name]
        return cls(self.config)


def build_population(config):
    """Convenience: build and return a population for given config."""
    return PopulationBuilder(config).build()