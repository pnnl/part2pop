from __future__ import annotations

import os
from pathlib import Path
from importlib.resources import files, as_file
import sys as _sys

# __all__ = ["data_path", "get_data_path"]

# def _discover_datasets_dir() -> Path | None:
#     """
#     Return a real filesystem Path to the packaged 'datasets' dir if present.
#     Works for wheels/zip installs and editable installs.
#     """
#     # Use the actual imported top-level module object (rename/case safe)
#     pkg_name = (__package__ or "part2pop").split(".", 1)[0]
#     pkg_obj = _sys.modules.get(pkg_name, pkg_name)
#     try:
#         ds = files(pkg_obj).joinpath("datasets")
#         if ds.is_dir():
#             with as_file(ds) as p:
#                 return Path(p)
#     except Exception:
#         pass
#     # Fallback for source checkouts
#     here = Path(__file__).resolve().parent
#     cand = here / "datasets"
#     return cand if cand.is_dir() else None

# def get_data_path() -> Path:
#     """
#     Preferred datasets directory. If user set part2pop_DATA_PATH, honor it.
#     Otherwise, return the packaged datasets path if available.
#     """
#     env = os.environ.get("part2pop_DATA_PATH")
#     if env:
#         return Path(env).expanduser()
#     ds = _discover_datasets_dir()
#     if ds:
#         return ds
#     # final fallback so we always return *something*
#     return Path.cwd() / "datasets"

# data_path: Path = get_data_path()

# # Optional: make legacy env-based lookups "just work" without user config.
# # Only set if user didn't already define it.
# os.environ.setdefault("part2pop_DATA_PATH", str(data_path))


# # def get_data_path() -> Path:
# #     # Highest priority: explicit override
# #     if (p := os.environ.get("part2pop_DATA_PATH")):
# #         return Path(p).expanduser()

# #     # Packaged datasets inside the installed wheel
# #     ds = files(_pkg).joinpath("datasets")
# #     if ds.is_dir():
# #         # If callers need a filesystem path (e.g., for C libs), materialize it:
# #         with as_file(ds) as pth:
# #             return Path(pth)

# #     # Last resort for source checkouts
# #     from pathlib import Path as _P
# #     cand = _P(__file__).resolve().parent / "datasets"
# #     return cand

# # data_path = get_data_path()
# # __all__ = ["data_path", "get_data_path"]

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
