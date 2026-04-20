"""

"""
#from .dispatcher import compute_variable, list_variables, describe_variable
from .builder import build_variable
# Eagerly import factory modules so decorators and build functions are registered.
from importlib import import_module
import pkgutil, os
_pkg_path = os.path.join(os.path.dirname(__file__), "population", "factory")
for _, module_name, _ in pkgutil.iter_modules([_pkg_path]):
  if module_name in ("__init__", "registry"):
    continue
  import_module(f"{__package__}.population.factory.{module_name}")


__all__ = [
  "build_variable",
  "compute_variable"
  "lists_variables"
  "describe_variable"]