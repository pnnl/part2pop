# viz/factory/registry.py
import importlib
import pkgutil
import os

_registry = {}

# fixme: the registry pattern is duplicated across modules
def register(name):
    def decorator(cls_or_fn):
        _registry[name] = cls_or_fn
        return cls_or_fn
    return decorator


def discover_plotter_types():
    """Discover all population type modules in the types/ submodule."""
    types_pkg = __package__  # The current package
    types_path = os.path.dirname(__file__)
    plotter_types = {}
    for _, module_name, _ in pkgutil.iter_modules([types_path]):
        module = importlib.import_module(f"{types_pkg}.{module_name}")
        if hasattr(module, "build") and callable(getattr(module, "build")):
            plotter_types[module_name] = module.build

    return plotter_types
