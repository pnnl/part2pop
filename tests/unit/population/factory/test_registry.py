# tests/unit/population/factory/test_registry.py

import pkgutil

from pyparticle.population.factory import registry as factory_registry
from pyparticle.population import factory as factory_pkg


def test_discover_population_types_covers_all_factory_modules():
    """
    Ensure that discover_population_types finds a build function for every
    population factory module under pyparticle.population.factory except
    the registry module itself.

    This guarantees that adding a new <module>.py with a build() function
    automatically makes it discoverable by the builder.
    """
    types = factory_registry.discover_population_types()
    assert isinstance(types, dict)
    for name, fn in types.items():
        assert callable(fn)

    # All .py modules under pyparticle.population.factory except "registry"
    module_names = {
        name
        for _, name, _ in pkgutil.iter_modules(factory_pkg.__path__)
        if not name.startswith("_") and name != "registry"
    }

    # discover_population_types returns keys that should match module names
    # (binned_lognormals, sampled_lognormals, monodisperse, mam4, partmc, ...)
    discovered = set(types.keys())

    # If you ever add a factory module without a build() function, this test
    # will fail and remind you to either add build() or explicitly ignore it.
    assert module_names == discovered
