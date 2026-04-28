# viz/factory/registry.py
import importlib
import pkgutil
import os
import warnings

_registry = {}

# fixme: the registry pattern is duplicated across modules
def register(name):
    def decorator(cls_or_fn):
        _registry[name] = cls_or_fn
        return cls_or_fn
    return decorator


def discover_plotter_types():
    """Discover plotter builders with decorator-first and safe fallback discovery."""
    types_pkg = __package__
    types_path = os.path.dirname(__file__)
    plotter_types = dict(_registry)
    for _, module_name, _ in pkgutil.iter_modules([types_path]):
        if module_name in {"registry", "__init__"} or module_name.startswith("_"):
            continue
        try:
            module = importlib.import_module(f"{types_pkg}.{module_name}")
        except Exception as exc:
            warnings.warn(
                f"Skipping viz factory module '{module_name}' during discovery: {exc}",
                RuntimeWarning,
            )
            continue
        if hasattr(module, "build") and callable(getattr(module, "build")):
            plotter_types.setdefault(module_name, module.build)
        if _registry:
            plotter_types.update(_registry)

    return plotter_types


def list_plotter_types():
    return sorted(discover_plotter_types().keys())
