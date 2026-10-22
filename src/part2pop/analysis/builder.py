from __future__ import annotations
from typing import Dict, Any, Callable

# fixme: shouldn't resolve_name, _ALIASES be part of e.g., global_registry.py?
from .population.factory.registry import resolve_name, _ALIASES
from .particle.factory.registry import resolve_particle_name
from .defaults import get_defaults_for_variable as _get_defaults_for_var
import warnings
# Unified builder: routes to population or particle registries

def _get_registry_builder(scope: str) -> Callable[[str], Callable]:
	"""Return a function that, given a variable name, returns the builder callable.

	scope: either 'population' or 'particle'
	The returned callable has signature get_builder(name: str) -> callable
	where the callable is a builder that when called with cfg returns the variable instance.
	"""
	if scope == "population":
		from .population.factory.registry import get_population_builder as _g
		return _g
	if scope == "particle":
		from .particle.factory.registry import get_particle_builder as _g
		return _g
	raise ValueError(f"Unknown scope '{scope}'")

def _merge_grid_layer(merged, layer, name):
    """Apply aliases at this layer's priority, without changing explicit ties."""
    layer = dict(layer)
    for canonical, alias in (("s_grid", "s_eval"), ("wvl_grid", "wvls")):
        if canonical in layer:
            layer.setdefault(alias, layer[canonical])
        elif alias in layer:
            layer[canonical] = layer[alias]
    distributions = {
        "dNdlnD": ("diam_grid", {"N_bins", "D_min", "D_max"}),
        "INSA_distribution": ("insa_grid", {"N_bins", "INSA_min", "INSA_max"}),
    }
    if name in distributions:
        centers, controls = distributions[name]
        has_centers = layer.get(centers) is not None
        has_edges = layer.get("edges") is not None
        if has_centers and not has_edges:
            merged.pop("edges", None)
        elif has_edges and not has_centers:
            merged.pop(centers, None)
        elif controls.intersection(layer) and not has_centers and not has_edges:
            merged.pop(centers, None)
            merged.pop("edges", None)
    merged.update(layer)


class VariableBuilder:
	"""Unified VariableBuilder that can build population or particle variables.

	Usage:
	  VariableBuilder(name, cfg=None, scope='population').build()
	"""
	def __init__(self, name: str, cfg: Dict[str, Any] | None = None, scope: str = "population"):
		user_requested = name
		if scope == 'population':
			canon = resolve_name(name)
		elif scope == 'particle':
			canon = resolve_particle_name(name)
		if user_requested != canon and user_requested in _ALIASES:
			warnings.warn(
				f"Variable alias '{user_requested}' is deprecated; use '{canon}' instead.",
				DeprecationWarning,
				stacklevel=2,
			)
		self.name = canon
		self.cfg = cfg or {}
		self._mods: Dict[str, Any] = {}
		self.scope = scope

	def modify(self, **k):
		# Policy decision deferred: repeated calls share one runtime layer.
		# Same-layer precedence still applies, so a later alias/bin request
		# need not replace an earlier canonical/center grid. Clear obsolete
		# grid keys explicitly; last-call-wins across keys needs an API decision.
		self._mods.update(k)
		return self

	def build(self):
		get_builder = _get_registry_builder(self.scope)
		builder: Callable = get_builder(self.name)
		# Attempt to get default config from builder.meta or from an instance
		defaults: Dict[str, Any] = {}
		if hasattr(builder, "meta"):
			defaults = dict(getattr(builder, "meta").default_cfg)
		else:
			try:
				inst = builder({})
				defaults = dict(getattr(inst, "meta").default_cfg)
			except Exception:
				defaults = {}

		# Merge order (lowest -> highest precedence):
		# 1) canonical defaults for this variable name (from analysis.defaults)
		# 2) builder/variable meta.default_cfg (variable-local defaults)
		# 3) user-supplied cfg passed to the builder
		# 4) runtime modifications via .modify()
		# fetch defaults for the canonical variable name
		try:
			canonical_defaults = _get_defaults_for_var(self.name)
			if canonical_defaults is None:
				canonical_defaults = {}
		except Exception:
			# be conservative: ignore defaults lookup failures
			canonical_defaults = {}
		merged = {}
		for layer in (canonical_defaults, defaults, self.cfg, self._mods):
			_merge_grid_layer(merged, layer, self.name)

		# Call the builder with merged cfg
		obj = builder(merged)
		return obj

def build_variable(name: str, scope: str = "population", var_cfg: dict | None = None):
    return VariableBuilder(name, var_cfg, scope=scope).build()

__all__ = ["VariableBuilder", "build_variable"]
