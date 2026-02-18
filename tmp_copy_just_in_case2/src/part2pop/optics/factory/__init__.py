"""Compatibility adapter for optics morphology factory.

This small shim exposes a simple `create_optical_particle` function and
`MORPHOLOGY_REGISTRY` accessor so older call sites that expect a
`part2pop.optics.factory` symbol can keep working while the richer
builder/discovery API lives in `part2pop.optics.builder` and
`part2pop.optics.factory.registry`.

Keep this file minimal to avoid changing package import semantics elsewhere.
"""
from typing import Any, Dict

from .registry import discover_morphology_types

try:
    # Import Particle to construct a base particle when callers pass species+masses
    from ...aerosol_particle import Particle
except Exception:
    Particle = None

try:
    # Helper to convert species names to objects if needed
    from ...species.registry import get_species
except Exception:
    get_species = None


def create_optical_particle(morphology: str, species, masses, **kwargs) -> Any:
    """
    Build an optical particle instance for the requested morphology.

    Parameters
    ----------
    morphology : str
        Morphology name (e.g., 'homogeneous', 'core_shell', 'fractal').
    species : sequence
        Sequence of species objects or species names.
    masses : sequence
        Sequence of masses corresponding to species (SI units).
    kwargs : dict
        Additional options forwarded into the morphology factory (rh_grid, wvl_grid, temp, etc.).

    Returns
    -------
    optical_particle : object
        Instance constructed by the morphology builder (must implement compute_optics etc.).
    """
    types = discover_morphology_types()
    if morphology not in types:
        raise ValueError(f"Unknown morphology type: {morphology}")
    cls_or_factory = types[morphology]

    # Ensure species objects
    species_objs = []
    for s in species:
        if hasattr(s, "name"):
            species_objs.append(s)
        else:
            if get_species is None:
                raise RuntimeError("Species registry not available to resolve species names")
            species_objs.append(get_species(str(s)))

    # Construct a base Particle-like object if Particle class is available
    base_particle = None
    if Particle is not None:
        base_particle = Particle(species=species_objs, masses=masses)
    else:
        # Fallback: provide a minimal object with required attributes
        class BaseLike:
            def __init__(self, species, masses):
                self.species = species
                self.masses = masses
        base_particle = BaseLike(species_objs, masses)

    # Expect class_or_factory to accept (base_particle, config)
    config = dict(kwargs)
    return cls_or_factory(base_particle, config)


def MORPHOLOGY_REGISTRY() -> Dict[str, Any]:
    """Return the discovered morphology mapping (name -> builder callable).
    This is a snapshot; callers should re-call if dynamic discovery is required.
    """
    return discover_morphology_types()
