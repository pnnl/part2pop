# viz/factory/registry.py
import importlib
import pkgutil
import os
import warnings

_registry = {}
_DISCOVERED = False

# fixme: the registry pattern is duplicated across modules
def register(name):
    def decorator(cls_or_fn):
        _registry[name] = cls_or_fn
        return cls_or_fn
    return decorator


def discover_plotter_types():
    """Discover plotter builders with decorator-first and safe fallback discovery.

    Results are cached after the first call; subsequent calls return immediately
    without re-scanning the filesystem.
    """
    global _DISCOVERED
    if _DISCOVERED:
        return dict(_registry)
    types_pkg = __package__
    types_path = os.path.dirname(__file__)
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
            _registry.setdefault(module_name, module.build)
    _DISCOVERED = True
    return dict(_registry)


def list_plotter_types():
    return sorted(discover_plotter_types().keys())


def describe_plotter_type(name):
    types = discover_plotter_types()
    if name not in types:
        available = ", ".join(sorted(types.keys())) or "<none>"
        raise ValueError(f"Unknown plotter type: {name}. Available types: {available}")
    builder = types[name]
    return {
        "name": name,
        "module": getattr(builder, "__module__", None),
        "type": getattr(builder, "__name__", type(builder).__name__),
        "description": (getattr(builder, "__doc__", None) or "").strip() or None,
        "defaults": getattr(builder, "defaults", None),
    }
