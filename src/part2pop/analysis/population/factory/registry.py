from __future__ import annotations
import importlib, pkgutil, os, difflib
import importlib.util
from typing import Dict, Callable, Iterable

class UnknownVariableError(KeyError):
    def __init__(self, name: str, suggestions: Iterable[str] | None = None):
        msg = f"Unknown variable '{name}'"
        if suggestions:
            msg += f". Did you mean: {', '.join(suggestions)}?"
        super().__init__(msg)

_REGISTRY: Dict[str, Callable] = {}
_ALIASES: Dict[str, str] = {}
_DISCOVERED = False

def register_variable(name: str):
    def deco(cls_or_fn: Callable):
        if name in _REGISTRY:
            raise KeyError(f"Variable '{name}' already registered")
        def _build(cfg=None):
            cfg = cfg or {}
            if isinstance(cls_or_fn, type):
                inst = cls_or_fn(cfg)
                return inst
            return cls_or_fn(cfg)

        _REGISTRY[name] = _build
        meta = getattr(cls_or_fn, "meta", None)
        if meta and getattr(meta, "aliases", None):
            for a in meta.aliases:
                if a in _ALIASES:
                    raise KeyError(f"Alias '{a}' already registered")
                _ALIASES[a] = name
        return cls_or_fn
    return deco

def _discover():
    global _DISCOVERED
    if _DISCOVERED:
        return
    pkg_name = __package__ or ".".join(__name__.split(".")[:-1])
    pkg_path = os.path.dirname(__file__)

    def _safe_import_module(fullname: str, file_path: str = None):
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

    for _, module_name, _ in pkgutil.iter_modules([pkg_path]):
        if module_name in ("registry", "__init__"):
            continue
        if module_name.startswith("_"):
            continue

        fullname = f"{pkg_name}.{module_name}"
        file_path = os.path.join(pkg_path, f"{module_name}.py")
        try:
            module = _safe_import_module(fullname, file_path=file_path)
        except Exception:
            continue

        if hasattr(module, "build") and callable(getattr(module, "build")):
            _REGISTRY[module_name] = getattr(module, "build")
    _DISCOVERED = True

def resolve_name(name: str) -> str:
    _discover()
    if name in _REGISTRY:
        return name
    if name in _ALIASES:
        return _ALIASES[name]
    suggestions = difflib.get_close_matches(name, list(_REGISTRY.keys()) + list(_ALIASES.keys()), n=3, cutoff=0.5)
    raise UnknownVariableError(name, suggestions)

def get_population_builder(name: str) -> Callable:
    return _REGISTRY[resolve_name(name)]

def list_variables(include_aliases: bool = False):
    _discover()
    names = sorted(_REGISTRY.keys())
    if include_aliases:
        names += sorted(_ALIASES.keys())
    return names

def describe_variable(name: str):
    builder = get_population_builder(name)
    meta = None
    if hasattr(builder, "meta"):
        meta = getattr(builder, "meta")
    else:
        try:
            inst = builder({})
            meta = getattr(inst, "meta", None)
        except Exception:
            meta = None

    if not meta:
        raise UnknownVariableError(name, suggestions=None)

    return {
        "name": meta.name,
        "value_key": meta.name,
        "axis_keys": list(meta.axis_names),
        "description": meta.description,
        "aliases": list(meta.aliases),
        "defaults": dict(meta.default_cfg),
        "units": dict(meta.units) if getattr(meta, "units", None) else None,
    }
