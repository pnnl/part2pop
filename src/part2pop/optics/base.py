"""Optical particle and population base classes.

Defines abstract `OpticalParticle` interface and `OpticalPopulation` which
aggregates per-particle cross-sections into population-level optical
coefficients (extinction, scattering, absorption, asymmetry).
"""

from abc import abstractmethod
import numpy as np

from ..aerosol_particle import Particle
from ..population.base import ParticlePopulation
# from .. import data_path

# NEW: use the refactored RI helper
from .refractive_index import build_refractive_index


class OpticalParticle(Particle):
    """
    Base class for all optical particle morphologies.

    Responsibilities:
    - set grids (rh_grid, wvl_grid, temp)
    - attach wavelength-aware refractive index to each species
    - allocate per-particle C* arrays (shape: N_rh x N_wvl)
    - provide default accessors for cross sections / indices
    """

    def __init__(self, base_particle, config, species_modifications={}):
        # copy species + masses
        super().__init__(species=base_particle.species, masses=base_particle.masses)

        # Grids (internal units: meters for wavelength)
        self.rh_grid = np.asarray(config.get("rh_grid", [0.0]), dtype=float)
        self.wvl_grid = np.asarray(config.get("wvl_grid", [550e-9]), dtype=float)
        self.temp = float(config.get("temp", 293.15))

        # FIXME: include this in config or make species_modification an optional input?
        # store species_modifications for possible per-morphology overrides
        self.species_modifications = config.get("species_modifications", {}) or {}

        # Allocate per-particle cross-section cubes (m²), shape (N_rh, N_wvl)
        N_rh = len(self.rh_grid)
        N_wvl = len(self.wvl_grid)
        self.Cabs = np.zeros((N_rh, N_wvl))
        self.Csca = np.zeros((N_rh, N_wvl))
        self.Cext = np.zeros((N_rh, N_wvl))
        self.g    = np.zeros((N_rh, N_wvl))
        
        self.Cabs_bc = np.zeros((N_rh, N_wvl))
        self.Csca_bc = np.zeros((N_rh, N_wvl))
        self.Cext_bc = np.zeros((N_rh, N_wvl))
        self.g_bc    = np.zeros((N_rh, N_wvl))

        # No debug printing here.

        # Attach refractive indices to each species once if not already present.
        # In the new flow, the optics population builder will usually attach RIs
        # once per species (preferred). This call is guarded so it becomes a
        # no-op when the population-level attachment ran first.
        self._attach_refractive_indices()
    
    def _attach_refractive_indices(self):
        """Attach a wavelength-aware refractive index object to each species."""
        # Allow a single "SOA" override to apply to common organic names
        soa_names = {'MSA','ARO1','ARO2','ALK1','OLE1','API1','API2','LIM1','LIM2'}
        soa_mods = self.species_modifications.get('SOA', {})

        for spec in self.species:
            # If species already has a wavelength-aware refractive_index attached
            # with the expected callables, don't rebuild it here.
            existing = getattr(spec, 'refractive_index', None)
            if existing is not None and hasattr(existing, 'real_ri_fun') and hasattr(existing, 'imag_ri_fun'):
                continue

            mods = self.species_modifications.get(spec.name, {})
            if not mods and spec.name in soa_names:
                mods = soa_mods  # inherit SOA envelope if specific override missing
            build_refractive_index(
                spec,
                self.wvl_grid,
                modifications=mods,
            )

    # --- abstract compute hook ---

    @abstractmethod
    def compute_optics(self):
        """Populate self.Cabs, self.Csca, self.Cext, self.g (shape: N_rh x N_wvl)."""

    # --- sensible defaults to reduce boilerplate in morphologies ---

    def get_cross_sections(self):
        """Return a dict of available cross-section arrays."""
        out = {"Cabs": self.Cabs, "Csca": self.Csca, "Cext": self.Cext, "g": self.g}
        # include variants if a morphology added them
        for k in ("Cabs_bc","Csca_bc","Cext_bc","g_bc",
                  "Cabs_clear","Csca_clear","Cext_clear","g_clear"):
            if hasattr(self, k) and getattr(self, k) is not None:
                out[k] = getattr(self, k)
        return out

    def get_refractive_indices(self):
        """Return a compact view of species RIs on the wavelength grid."""
        # Each species has .refractive_index with .real_ri_fun/.imag_ri_fun
        return [
            (
                s.name,
                s.refractive_index.real_ri_fun(self.wvl_grid),
                s.refractive_index.imag_ri_fun(self.wvl_grid),
            )
            for s in self.species
        ]

    def get_cross_section(self, optics_type: str, rh_idx=None, wvl_idx=None):
        """Convenience getter on the per-particle arrays (no aggregation)."""
        key = str(optics_type).lower().strip()
        src = None
        if   key in ("abs","cabs","absorption"): src = self.Cabs
        elif key in ("sca","csca","scattering"): src = self.Csca
        elif key in ("ext","cext","extinction"): src = self.Cext
        elif key in ("g","asym","asymmetry"):    src = self.g
        else:
            raise ValueError(f"Unknown per-particle optics_type: {optics_type}")

        # Fancy indexing guards
        arr = src
        if rh_idx is not None:
            arr = arr[rh_idx, :]
        if wvl_idx is not None:
            arr = arr[:, wvl_idx]
        return arr


class OpticalPopulation(ParticlePopulation):
    """
    Aggregate per-particle optical cross-sections (C*: m²) onto a common RH×λ grid.
    Aggregation to coefficients (m⁻¹) is done by summing C* × number_concentration (m⁻³).
    """

    def __init__(self, base_population, rh_grid, wvl_grid):
        self.base_population = base_population
        self.species = base_population.species
        self.spec_masses = base_population.spec_masses.copy()
        self.num_concs = base_population.num_concs.copy()
        self.ids = list(base_population.ids)

        self.rh_grid = np.asarray(rh_grid, dtype=float)
        self.wvl_grid = np.asarray(wvl_grid, dtype=float)

        nP = len(self.ids)
        nR = len(self.rh_grid)
        nW = len(self.wvl_grid)

        # main cubes: per particle
        self.Cabs  = np.zeros((nP, nR, nW))
        self.Csca  = np.zeros((nP, nR, nW))
        self.Cext  = np.zeros((nP, nR, nW))
        self.g     = np.zeros((nP, nR, nW))

        # optional variant cubes (filled by morphologies that provide them)
        self.Cabs_bc  = None; self.Csca_bc  = None; self.Cext_bc  = None; self.g_bc  = None
        self.Cabs_clear=None; self.Csca_clear=None; self.Cext_clear=None; self.g_clear=None

        # lazily computed
        self.tkappas = None
        self.shell_tkappas = None

    def _find_index(self, part_id):
        try:
            return self.ids.index(part_id)
        except ValueError:
            raise ValueError(f"Particle id {part_id} not found in population ids")

    def add_optical_particle(self, optical_particle, part_id):
        """
        Copy computed cross-sections from a per-particle 'optical_particle'
        into the population arrays. Ensures compute_optics() is called once.
        """
        if getattr(optical_particle, "Cext", None) is None or optical_particle.Cext.size == 0:
            optical_particle.compute_optics()

        i = self._find_index(part_id)

        self.Cabs[i, :, :] = optical_particle.Cabs
        self.Csca[i, :, :] = optical_particle.Csca
        self.Cext[i, :, :] = optical_particle.Cext
        self.g[i,   :, :]  = optical_particle.g

        # Variants (create on first use if present)
        for name in ("Cabs_bc","Csca_bc","Cext_bc","g_bc",
                     "Cabs_clear","Csca_clear","Cext_clear","g_clear"):
            if hasattr(optical_particle, name) and getattr(optical_particle, name) is not None:
                if getattr(self, name) is None:
                    shape = (len(self.ids), len(self.rh_grid), len(self.wvl_grid))
                    setattr(self, name, np.zeros(shape))
                getattr(self, name)[i, :, :] = getattr(optical_particle, name)

    def _select_indices(self, rh, wvl):
        """Return (rh_idx, wvl_idx) or (slice(None), slice(None)) if None."""
        if rh is None:
            rh_idx = slice(None)
        else:
            arr = np.asarray(self.rh_grid)
            hits = np.where(np.isclose(arr, rh))[0]
            if len(hits) == 0:
                raise ValueError(f"RH {rh} not found in rh_grid {self.rh_grid}")
            rh_idx = int(hits[0])

        if wvl is None:
            wvl_idx = slice(None)
        else:
            arr = np.asarray(self.wvl_grid)
            hits = np.where(np.isclose(arr, wvl))[0]
            if len(hits) == 0:
                raise ValueError(f"Wavelength {wvl} not found in wvl_grid {self.wvl_grid}")
            wvl_idx = int(hits[0])

        return rh_idx, wvl_idx
    
    # (keep the more featureful _safe_index_2d defined below)

    def _safe_index_2d(self, arr2d, i, j):
        """
        Robust 2D indexing that handles:
          - both slice(None): return full array
          - one int, one slice: return a 1D array
          - both ints: return scalar
          - if given sequences (list/ndarray) for both, use np.ix_ to get the grid
        """
        # Normalize Python range -> slice
        if isinstance(i, range):
            i = slice(i.start, i.stop, i.step)
        if isinstance(j, range):
            j = slice(j.start, j.stop, j.step)

        seq_types = (list, tuple, np.ndarray)

        if isinstance(i, seq_types) and isinstance(j, seq_types):
            i_idx = np.asarray(i, dtype=int)
            j_idx = np.asarray(j, dtype=int)
            return arr2d[np.ix_(i_idx, j_idx)]
        elif isinstance(i, seq_types):
            i_idx = np.asarray(i, dtype=int)
            return arr2d[i_idx, j]
        elif isinstance(j, seq_types):
            j_idx = np.asarray(j, dtype=int)
            return arr2d[i, j_idx]
        else:
            return arr2d[i, j]

    def get_optical_coeff(self, optics_type, rh=None, wvl=None):
        """
        Compute the population-level optical property.

        optics_type:
          - 'b_abs','absorption','abs'     -> sum_i Cabs_i * N_i
          - 'b_scat','scattering','scat'   -> sum_i Csca_i * N_i
          - 'b_ext','extinction','ext'     -> sum_i Cext_i * N_i
          - 'g','asymmetry'                -> scattering-weighted mean:
                                             sum_i (g_i * Csca_i * N_i) / sum_i (Csca_i * N_i)

        rh, wvl:
          - None means return values across the full grid dimension(s).
          - If both provided, return a scalar.
        """
        rh_idx, wvl_idx = self._select_indices(rh, wvl)
        
        key = str(optics_type).lower()
        w = self.num_concs.reshape(-1, 1, 1)  # weight by number concentration

        if key in ('b_abs', 'absorption', 'abs'):
            total = np.sum(self.Cabs * w, axis=0)
        elif key in ('b_scat', 'scattering', 'scat'):
            total = np.sum(self.Csca * w, axis=0)
        elif key in ('b_ext', 'extinction', 'ext'):
            total = np.sum(self.Cext * w, axis=0)
        elif key in ('g', 'asymmetry'):
            num = np.sum(self.g * self.Csca * w, axis=0)
            den = np.sum(self.Csca * w, axis=0)
            with np.errstate(invalid='ignore', divide='ignore'):
                total = np.where(den > 0, num / den, 0.0)
        else:
            raise ValueError(f"optics_type = {optics_type} not implemented.")
        
        #out = total[rh_idx, wvl_idx]
        out = self._safe_index_2d(total, rh_idx, wvl_idx)        
        if np.ndim(out) == 0:
            return float(out)
        return out

    def compute_effective_kappas(self):
        self.tkappas = np.zeros(len(self.ids))
        self.shell_tkappas = np.zeros(len(self.ids))
        for ii, pid in enumerate(self.ids):
            p = self.base_population.get_particle(pid)
            self.tkappas[ii] = float(p.get_tkappa())
            self.shell_tkappas[ii] = float(p.get_shell_tkappa())
    
