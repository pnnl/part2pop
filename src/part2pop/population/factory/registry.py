import importlib
import pkgutil
import os
import warnings

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

def discover_population_types():
    """Discover population builders with decorator-first and safe fallback discovery."""
    types_pkg = __package__
    types_path = os.path.dirname(__file__)
    population_types = dict(_registry)
    for _, module_name, _ in pkgutil.iter_modules([types_path]):
        if module_name in {"registry", "__init__"} or module_name.startswith("_"):
            continue
        try:
            module = importlib.import_module(f"{types_pkg}.{module_name}")
        except Exception as exc:
            warnings.warn(
                f"Skipping population factory module '{module_name}' during discovery: {exc}",
                RuntimeWarning,
            )
            continue
        if hasattr(module, "build") and callable(getattr(module, "build")):
            population_types.setdefault(module_name, module.build)
        if _registry:
            population_types.update(_registry)
    return population_types


def list_population_types():
    return sorted(discover_population_types().keys())
