"""Global registry router for analysis variables.

Provides a single entrypoint to resolve variable builders/classes across
multiple "families" (population vs particle). Mirrors the pattern used in
other projects where a top-level registry routes to family-specific
discovery functions.

Usage:
  from part2pop.analysis import global_registry as gr
  builder = gr.get_variable_builder('dNdlnD', {})
  inst = builder({})

This module intentionally does not eagerly import all family factories; it
defers to family discovery functions which may import factory modules on
first call.
"""
from __future__ import annotations
from typing import Callable, Dict


def _discover_population_variable_types() -> Dict[str, Callable]:
    """Discover population variables and return mapping full_name -> builder.

    full_name keys use the suffix convention produced by `family_to_suffix`.
    """
    from .population.factory import registry as preg

    # Ensure discovery has run
    try:
        preg._discover()
    except Exception:
        # fall back to calling list/get_builder
        pass

    suffix = family_to_suffix("PopulationVariable")
    out: Dict[str, Callable] = {}
    for name in preg.list_variables():
        full = name + suffix
        out[full] = preg.get_builder(name)
    return out


def _discover_particle_variable_types() -> Dict[str, Callable]:
    from .particle.factory import registry as parg

    try:
        parg._discover_particle_factories()
    except Exception:
        pass

    suffix = family_to_suffix("ParticleVariable")
    out: Dict[str, Callable] = {}
    for name in parg.list_particle_variables():
        full = name + suffix
        out[full] = parg.get_particle_builder(name)
    return out


FAMILY_DISCOVERY_FUNCS = {
    "PopulationVariable": _discover_population_variable_types,
    "ParticleVariable": _discover_particle_variable_types,
}

# Build DEFAULT_VARIABLE_FAMILIES by discovering available variables across
# families and mapping base variable names to their most likely family.
DEFAULT_VARIABLE_FAMILIES: Dict[str, str] = {}
for fam_name, discover_fn in FAMILY_DISCOVERY_FUNCS.items():
    try:
        builders = discover_fn()
    except Exception:
        builders = {}
    for full in builders.keys():
        # base name is part before the first dot
        base = full.split('.', 1)[0] if '.' in full else full
        if base not in DEFAULT_VARIABLE_FAMILIES:
            DEFAULT_VARIABLE_FAMILIES[base] = fam_name


def family_to_suffix(family_name: str) -> str:
    """Convert a family name like 'PopulationVariable' to a suffix like '.population'.

    Expects family names ending in 'Variable'.
    """
    if not family_name.endswith("Variable"):
        raise ValueError("family_name must end with 'Variable'")
    core = family_name[: -len("Variable")]
    return "." + core.lower()


def build_full_variable_name(var_name: str, family_name: str) -> str:
    if "." in var_name:
        return var_name
    return var_name + family_to_suffix(family_name)


def get_variable_builder(name: str, config: Dict | None = None) -> Callable:
    """Resolve a variable name to a builder callable across families.

    Args:
      name: variable name (may be fully-qualified with suffix)
      config: optional dict; may include 'family' to override default

    Returns:
      builder callable (signature build(cfg=None) -> instance)

    Raises ValueError if resolution fails.
    """
    config = config or {}
    family = config.get("family") or DEFAULT_VARIABLE_FAMILIES.get(name)
    # Lazily populate DEFAULT_VARIABLE_FAMILIES if empty
    if not family and not DEFAULT_VARIABLE_FAMILIES:
        for fam_name, discover_fn in FAMILY_DISCOVERY_FUNCS.items():
            try:
                builders = discover_fn()
            except Exception:
                builders = {}
            for full in builders.keys():
                base = full.split('.', 1)[0] if '.' in full else full
                if base not in DEFAULT_VARIABLE_FAMILIES:
                    DEFAULT_VARIABLE_FAMILIES[base] = fam_name
        family = config.get("family") or DEFAULT_VARIABLE_FAMILIES.get(name)
    if not family:
        # If user provided a fully-qualified name we can infer family from suffix
        if "." in name:
            suffix = name.split(".", 1)[1]
            # map suffix back to a family if possible
            for fam in FAMILY_DISCOVERY_FUNCS:
                if family_to_suffix(fam).lstrip('.') == suffix:
                    family = fam
                    break
    if not family:
        raise ValueError(f"No family specified and no default found for variable '{name}'")

    discover_fn = FAMILY_DISCOVERY_FUNCS.get(family)
    if not discover_fn:
        raise ValueError(f"Unknown family '{family}'")

    full_name = build_full_variable_name(name, family)
    builders = discover_fn()
    if full_name not in builders:
        raise ValueError(f"Variable '{full_name}' not found in family '{family}'")
    return builders[full_name]


def get_variable_class(name: str, config: Dict | None = None) -> type:
    """Return the variable class (builder result type) for the named variable.

    This calls `get_variable_builder` and attempts to instantiate a no-arg
    builder to access its class. If the builder is a plain factory function,
    it should return an instance when called with empty cfg.
    """
    builder = get_variable_builder(name, config=config)
    try:
        inst = builder({})
        return type(inst)
    except Exception as e:
        raise RuntimeError("Failed to instantiate variable builder to get class") from e


def list_registered_variables(include_aliases: bool = False) -> Dict[str, str]:
    """Return a mapping of full_name -> family for all registered variables.

    include_aliases toggles whether alias names are included (when families
    support aliases via their list functions).
    """
    out: Dict[str, str] = {}
    for fam, fn in FAMILY_DISCOVERY_FUNCS.items():
        builders = fn()
        for full_name in builders.keys():
            out[full_name] = fam
    return out


__all__ = [
    "FAMILY_DISCOVERY_FUNCS",
    "DEFAULT_VARIABLE_FAMILIES",
    "family_to_suffix",
    "build_full_variable_name",
    "get_variable_builder",
    "get_variable_class",
    "list_registered_variables",
]
