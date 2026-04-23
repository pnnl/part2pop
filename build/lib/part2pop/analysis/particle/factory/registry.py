from __future__ import annotations
import importlib, pkgutil, os, difflib
import importlib.util
from typing import Dict, Callable, Iterable
from ..base import ParticleVariable


_PARTICLE_REG: Dict[str, Callable] = {}
_PARTICLE_ALIASES: Dict[str, str] = {}
_PARTICLE_DISCOVERED = False


class UnknownParticleVariableError(KeyError):
    def __init__(self, name: str, suggestions: Iterable[str] | None = None):
        msg = f"Unknown particle variable '{name}'"
        if suggestions:
            msg += f". Did you mean: {', '.join(suggestions)}?"
        super().__init__(msg)


def register_particle_variable(name: str):
    def deco(cls_or_fn: Callable):
        if name in _PARTICLE_REG:
            raise KeyError(f"Particle variable '{name}' already registered")

        def _build(cfg=None):
            cfg = cfg or {}
            if isinstance(cls_or_fn, type):
                return cls_or_fn(cfg)
            return cls_or_fn(cfg)

        _PARTICLE_REG[name] = _build
        meta = getattr(cls_or_fn, "meta", None)
        if meta and getattr(meta, "aliases", None):
            for a in meta.aliases:
                if a in _PARTICLE_ALIASES:
                    raise KeyError(f"Alias '{a}' already registered")
                _PARTICLE_ALIASES[a] = name
        return cls_or_fn

    return deco


def _discover_particle_factories():
    global _PARTICLE_DISCOVERED
    if _PARTICLE_DISCOVERED:
        return
    pkg_path = os.path.dirname(__file__)
    # factory modules live in the same folder as this registry
    for _, module_name, _ in pkgutil.iter_modules([pkg_path]):
        if module_name in ("registry", "__init__"):
            continue
        if module_name.startswith("_"):
            continue
        fullname = f"{__package__}.{module_name}"
        file_path = os.path.join(pkg_path, f"{module_name}.py")
        try:
            module = importlib.import_module(fullname)
        except Exception:
            try:
                spec = importlib.util.spec_from_file_location(fullname, file_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                else:
                    continue
            except Exception:
                continue

        if hasattr(module, "build") and callable(getattr(module, "build")):
            _PARTICLE_REG[module_name] = getattr(module, "build")

    _PARTICLE_DISCOVERED = True


def resolve_particle_name(name: str) -> str:
    _discover_particle_factories()
    if name in _PARTICLE_REG:
        return name
    if name in _PARTICLE_ALIASES:
        return _PARTICLE_ALIASES[name]
    suggestions = difflib.get_close_matches(name, list(_PARTICLE_REG.keys()) + list(_PARTICLE_ALIASES.keys()), n=3, cutoff=0.5)
    raise UnknownParticleVariableError(name, suggestions)


def get_particle_builder(name: str) -> Callable:
    return _PARTICLE_REG[resolve_particle_name(name)]


def list_particle_variables(include_aliases: bool = False):
    _discover_particle_factories()
    names = sorted(_PARTICLE_REG.keys())
    if include_aliases:
        names += sorted(_PARTICLE_ALIASES.keys())
    return names


def describe_particle_variable(name: str):
    builder = get_particle_builder(name)
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
        raise UnknownParticleVariableError(name, suggestions=None)

    return {
        "name": meta.name,
        "value_key": meta.name,
        "axis_keys": list(meta.axis_names),
        "description": meta.description,
        "aliases": list(meta.aliases),
        "defaults": dict(meta.default_cfg),
        "units": dict(meta.units) if getattr(meta, "units", None) else None,
    }

