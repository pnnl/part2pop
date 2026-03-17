import importlib
import importlib.util
import pkgutil
import os

_morphology_registry = {}

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
    """
    # Start with anything already registered
    types = dict(_morphology_registry)

    pkg_name = __package__ or ".".join(__name__.split(".")[:-1])
    pkg_path = os.path.dirname(__file__)

    # Iterate modules present under this package directory
    for _, module_name, _ in pkgutil.iter_modules([pkg_path]):
        # Skip this registry module itself to avoid odd re-import loops
        if module_name in ("registry", "__init__"):
            continue

        fullname = f"{pkg_name}.{module_name}"
        file_path = os.path.join(pkg_path, f"{module_name}.py")
        try:
            module = _safe_import_module(fullname, file_path=file_path)
        except Exception:
            # Skip broken modules but continue discovery of others
            continue

        # If module exposes a build callable, include it by module name.
        # This callable must accept (base_particle, config) in this design.
        if hasattr(module, "build") and callable(getattr(module, "build")):
            types[module_name] = module.build

        # Merge any new entries registered by import side-effects
        if _morphology_registry:
            types.update(_morphology_registry)

    return types