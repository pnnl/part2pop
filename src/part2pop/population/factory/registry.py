import importlib
import pkgutil
import os

_registry = {}

def register(name):
    def decorator(cls_or_fn):
        _registry[name] = cls_or_fn
        return cls_or_fn
    return decorator

# def discover_population_types():
#     types_pkg = __package__
#     types_path = os.path.dirname(__file__)

#     # Import modules one-by-one, safely
#     for _, module_name, _ in pkgutil.iter_modules([types_path]):
#         try:
#             importlib.import_module(f"{types_pkg}.{module_name}")
#         except Exception as e:
#             # Optional: warn, but DO NOT crash discovery
#             # print(f"Skipping population factory '{module_name}': {e}")
#             continue

#     # Prefer decorator-based registry
#     if _registry:
#         return dict(_registry)

#     # Fallback: filename-based
#     population_types = {}
#     for _, module_name, _ in pkgutil.iter_modules([types_path]):
#         try:
#             module = importlib.import_module(f"{types_pkg}.{module_name}")
#         except Exception:
#             continue
#         if hasattr(module, "build"):
#             population_types[module_name] = module.build

#     return population_types

# #FIXME: remove this once we are sure decorator-based registry is stable
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
