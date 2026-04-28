import importlib
import importlib.util
import pkgutil
import os
import warnings

_morphology_registry = {}
_DISCOVERED = False

def register(name):
    """
    Decorator for morphology builders/classes that can be called as (base_particle, config).
    """
    def decorator(cls_or_fn):
        _morphology_registry[name] = cls_or_fn
        return cls_or_fn
    return decorator

def _safe_import_module(fullname: str, file_path: str = None):
    """
    Try importing a module by name; if that fails and file_path is provided,
    import it from the file path directly.
    """
    try:
        return importlib.import_module(fullname)
    except ModuleNotFoundError:
        if not file_path:
            raise
        spec = importlib.util.spec_from_file_location(fullname, file_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        raise

def discover_morphology_types():
    """
    Discover morphology builders that live in THIS package (factory folder).
    Returns a dict: name -> builder/class callable (from @register) or module.build fallback.

    Results are cached after the first call; subsequent calls return immediately
    without re-scanning the filesystem.
    """
    global _DISCOVERED
    if _DISCOVERED:
        return dict(_morphology_registry)

    pkg_name = __package__ or ".".join(__name__.split(".")[:-1])
    pkg_path = os.path.dirname(__file__)

    # Iterate modules present under this package directory
    for _, module_name, _ in pkgutil.iter_modules([pkg_path]):
        # Skip this registry module itself to avoid odd re-import loops
        if module_name in ("registry", "__init__") or module_name.startswith("_"):
            continue

        fullname = f"{pkg_name}.{module_name}"
        file_path = os.path.join(pkg_path, f"{module_name}.py")
        try:
            module = _safe_import_module(fullname, file_path=file_path)
        except Exception as exc:
            warnings.warn(
                f"Skipping freezing factory module '{module_name}' during discovery: {exc}",
                RuntimeWarning,
            )
            continue

        # If module exposes a build callable, include it by module name.
        # Decorator-registered entries (already in _morphology_registry) take priority.
        if hasattr(module, "build") and callable(getattr(module, "build")):
            _morphology_registry.setdefault(module_name, module.build)

    _DISCOVERED = True
    return dict(_morphology_registry)


def list_freezing_types():
    return sorted(discover_morphology_types().keys())


def describe_freezing_type(name):
    types = discover_morphology_types()
    if name not in types:
        available = ", ".join(sorted(types.keys())) or "<none>"
        raise ValueError(f"Unknown freezing morphology type: {name}. Available types: {available}")
    builder = types[name]
    return {
        "name": name,
        "module": getattr(builder, "__module__", None),
        "type": getattr(builder, "__name__", type(builder).__name__),
        "description": (getattr(builder, "__doc__", None) or "").strip() or None,
        "defaults": getattr(builder, "defaults", None),
    }
