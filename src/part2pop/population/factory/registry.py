import importlib
import pkgutil
import os

_registry = {}

def register(name):
    def decorator(cls_or_fn):
        _registry[name] = cls_or_fn
        return cls_or_fn
    return decorator

def discover_population_types():
    """Discover all population type modules in the types/ submodule."""
    types_pkg = __package__  # The current package
    types_path = os.path.dirname(__file__)
    population_types = {}
    for _, module_name, _ in pkgutil.iter_modules([types_path]):
        module = importlib.import_module(f"{types_pkg}.{module_name}")
        if hasattr(module, "build"):
            population_types[module_name] = module.build
    return population_types