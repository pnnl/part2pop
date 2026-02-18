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
		
		merged = dict(defaults)
		# Merge order (lowest -> highest precedence):
		# 1) canonical defaults for this variable name (from analysis.defaults)
		# 2) builder/variable meta.default_cfg (variable-local defaults)
		# 3) user-supplied cfg passed to the builder
		# 4) runtime modifications via .modify()
		merged = {}
		# fetch defaults for the canonical variable name
		try:
			merged.update(_get_defaults_for_var(self.name))
		except Exception:
			# be conservative: ignore defaults lookup failures
			pass
		# overlay variable-local defaults then user cfg and runtime mods
		merged.update(defaults)
		merged.update(self.cfg)
		merged.update(self._mods)
		
		# Call the builder with merged cfg
		obj = builder(merged)
		return obj

def build_variable(name: str, scope: str = "population", var_cfg={}):	
    return VariableBuilder(name, var_cfg, scope=scope).build()


__all__ = ["VariableBuilder", "build_variable"]
