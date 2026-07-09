from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _pkg_version

import os
from pathlib import Path
from importlib.resources import files, as_file
import sys as _sys

# Public helpers
from .utilities import get_number

from .aerosol_particle import Particle, make_particle, make_particle_from_masses

# Updated imports for new species/registry structure
from .species.base import AerosolSpecies
from .species.registry import (
    get_species,
    register_species,
    list_species,
    extend_species,
    retrieve_one_species,
)

from .population.base import ParticlePopulation
from .population import build_population

from .optics.builder import build_optical_particle, build_optical_population

try:
    __version__ = _pkg_version("part2pop")
except PackageNotFoundError:
    __version__ = "1.1.0"
