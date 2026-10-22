# optics/factory/homogeneous.py
import numpy as np
import math
import warnings

from .registry import register
from ..base import OpticalParticle
from ..refractive_index import build_refractive_index

try:
    from part2pop._patch import patch_pymiescatt
    patch_pymiescatt()
    from PyMieScatt import MieQ
    _PMS_ERR = None
except Exception as e:
    MieQ = None
    _PMS_ERR = e


# Rayleigh theory is an asymptotic small-particle approximation. Keep its
# opt-in fallback restricted to a deliberately conservative x << 1 regime.
# x = 2*pi*r/lambda < 0.1 means radius < lambda / (20*pi): about 8.8 nm at
# 550 nm and 6.4 nm at 400 nm. At x = 0.1 Rayleigh agrees with Mie to within
# about 0.5% (scattering) and 1.2% (absorption) even for strongly absorbing
# BC-like m = 1.95 + 0.79i, so the bound is conservative.
_RAYLEIGH_MAX_X = 0.1
# Conservative, empirically checked acceptance policy; see the validity doc.
# This bounds supported inputs, not a universal continuum error guarantee.
_RAYLEIGH_REAL_RANGE = (1.0, 2.0)
_RAYLEIGH_IMAG_RANGE = (0.0, 1.0)
# Mixing indices at an inclusive bound can overshoot it by a few ULPs.
# Allow only float64 roundoff on positive bounds; negative absorption stays
# unsupported even when its magnitude is smaller than this relative allowance.
_RAYLEIGH_RI_RTOL = 8.0 * np.finfo(float).eps

@register("homogeneous")
class HomogeneousParticle(OpticalParticle):
    """
    Homogeneous sphere morphology optical particle model with RH and wavelength dependence.

    Constructor expects (base_particle, config) to align with the factory builder.
    Rayleigh fallback must be explicitly enabled with
    ``allow_rayleigh_fallback=True`` and is restricted to x < 0.1 at every
    evaluated RH and wavelength (wet radius below about 8.8 nm at 550 nm). One
    larger particle anywhere on the grid aborts construction with ValueError.
    The fallback additionally requires finite 1 <= Re(m) <= 2 and
    0 <= Im(m) <= 1 at every evaluated point. This is an empirically checked
    acceptance policy; see docs/rayleigh-fallback-validity.md.

    Optional config (read by OpticalParticle or here):
      - rh_grid, wvl_grid, temp (K), specdata_path, species_modifications
      - allow_rayleigh_fallback (boolean, default False; Rayleigh opt-in)

    A legacy single_scatter_albedo attribute is retained but unused by both
    the Mie and Rayleigh calculations. It does not select a fallback.

    If PyMieScatt is unavailable, construction fails unless the explicit
    Rayleigh fallback is enabled.
    """

    def __init__(self, base_particle, config):
        super().__init__(base_particle, config)
        self.allow_rayleigh_fallback = bool(config.get("allow_rayleigh_fallback", False))

        # Refractive indices are attached at the population level by the
        # optics builder; the base class's _attach_refractive_indices is
        # guarded and will no-op if the species already have wavelength-aware
        # RIs. Keep the call to the base preparation intact.

        # Retain the legacy SSA attribute for compatibility only; neither the
        # Mie nor Rayleigh calculations use it.
        self.single_scatter_albedo = float(config.get("single_scatter_albedo", 0.9))

        # Precompute geometry & per-wavelength dry/water RIs
        self._prepare_geometry_and_ris()

        # Do the optics
        self.compute_optics()

    def _prepare_geometry_and_ris(self):
        """
        Precompute:
          - dry particle volume
          - water volumes vs RH (from Dwet - Ddry)
          - wavelength-dependent RIs for dry mix and water
        """
        # Water volumes vs RH from Dwet(RH) and Ddry
        Ddry = float(self.get_Ddry())

        self.h2o_vols = np.zeros(len(self.rh_grid))
        for rr, rh in enumerate(self.rh_grid):
            Dw = float(self.get_Dwet(RH=float(rh), T=self.temp))
            self.h2o_vols[rr] = (math.pi / 6.0) * (Dw ** 3 - Ddry ** 3) if rh > 0.0 else 0.0

        # Per-wavelength complex RIs
        Nw = len(self.wvl_grid)
        self.dry_ris = np.zeros(Nw, dtype=complex)
        self.h2o_ris = np.zeros(Nw, dtype=complex)

        # Species partitioning (use Particle-provided helpers if available)
        vks = self.get_vks()  # species "dry" volumes for partitioning
        h2o_idx = self.idx_h2o()

        # Build dry mixture RI by volume-weighted averaging of all non-water species
        dry_indices = [ii for ii in range(len(self.species)) if ii != h2o_idx]
        self.dry_vol = np.sum([vks[ii] for ii in dry_indices])
        for ww in range(Nw):
            # fixme: should this go in some effective_ri helper?
            n_dry = 0.0
            k_dry = 0.0
            for ii in dry_indices:
                f = float(vks[ii] / self.dry_vol)
                n_dry += self.species[ii].refractive_index.real_ri_fun(self.wvl_grid[ww]) * f
                k_dry += self.species[ii].refractive_index.imag_ri_fun(self.wvl_grid[ww]) * f
            self.dry_ris[ww] = complex(n_dry, k_dry)
            # Water RI (always set)
            n_w = self.species[h2o_idx].refractive_index.real_ri_fun(self.wvl_grid[ww])
            k_w = self.species[h2o_idx].refractive_index.imag_ri_fun(self.wvl_grid[ww])
            self.h2o_ris[ww] = complex(n_w, k_w)
    
    def _mixture_ri(self, rr: int, ww: int) -> complex:
        """
        Volume-weighted homogeneous mixture of dry material and water at a given RH and wavelength.
        """
        v_h2o = self.h2o_vols[rr]
        v_dry = self.dry_vol
        if (v_h2o + v_dry) <= 0.0:
            return complex(1.0, 0.0)
        return (self.h2o_ris[ww] * v_h2o + self.dry_ris[ww] * v_dry) / (v_h2o + v_dry)

    @staticmethod
    def _validate_rayleigh_domain(m, wavelength_m, radius_m):
        if (not np.all(np.isfinite([wavelength_m, radius_m]))
                or wavelength_m <= 0 or radius_m < 0):
            raise ValueError("Rayleigh fallback geometry requires a finite positive wavelength "
                             "and finite nonnegative radius.")
        size_parameter = 2.0 * math.pi * radius_m / wavelength_m
        if not np.isfinite(size_parameter) or size_parameter >= _RAYLEIGH_MAX_X:
            raise ValueError(
                "Rayleigh fallback is outside its supported "
                f"small-particle regime: x={size_parameter:.3g} "
                f"must be < {_RAYLEIGH_MAX_X:g}."
            )
        real_min, real_max = _RAYLEIGH_REAL_RANGE
        imag_min, imag_max = _RAYLEIGH_IMAG_RANGE
        if (not np.all(np.isfinite([m.real, m.imag]))
                or not real_min * (1 - _RAYLEIGH_RI_RTOL) <= m.real <= real_max * (1 + _RAYLEIGH_RI_RTOL)
                or not imag_min <= m.imag <= imag_max * (1 + _RAYLEIGH_RI_RTOL)):
            raise ValueError(
                "Rayleigh fallback refractive index must satisfy "
                "1 <= Re(m) <= 2 and 0 <= Im(m) <= 1; use PyMieScatt "
                "for inputs outside this empirically checked domain."
            )

    @staticmethod
    def _rayleigh_cross_sections(m: complex, wavelength_m: float, radius_m: float):
        """Return Rayleigh Cext, Csca, Cabs, and g for a homogeneous sphere.

        This is the small-particle approximation and is most accurate when
        x = 2*pi*r/lambda << 1. The cross-sections use the standard Rayleigh
        homogeneous-sphere result with K = (m**2 - 1) / (m**2 + 2); see
        Bohren and Huffman, Absorption and Scattering of Light by Small
        Particles, Chapter 5.

        ``m = n + ik`` uses part2pop's convention of a non-negative imaginary
        part. Im(K) is then non-negative; a negative value (from k < 0, for
        example an RI given in the opposite sign convention) is clamped to zero
        absorption rather than producing a negative cross-section.
        """
        k = 2.0 * math.pi / wavelength_m
        m2 = m * m
        polarizability_factor = (m2 - 1.0) / (m2 + 2.0)

        csca = (
            (8.0 * math.pi / 3.0)
            * (k ** 4)
            * (radius_m ** 6)
            * abs(polarizability_factor) ** 2
        )
        cabs = (
            4.0
            * math.pi
            * k
            * (radius_m ** 3)
            * max(0.0, float(polarizability_factor.imag))
        )
        cext = csca + cabs

        # Rayleigh scattering is symmetric fore/aft, so <cos(theta)> = 0.
        return cext, csca, cabs, 0.0

    def compute_optics(self):
        """
        Compute cross-sections and asymmetry parameter per (RH, wavelength).

        PyMieScatt is the default solver. Rayleigh is used only when explicitly
        enabled and every evaluated size parameter remains below
        _RAYLEIGH_MAX_X.
        """
        if MieQ is None:
            if not self.allow_rayleigh_fallback:
                error = ImportError(
                    "PyMieScatt is required for homogeneous optics. "
                    "Set allow_rayleigh_fallback=True only to opt into the "
                    f"Rayleigh approximation for x < {_RAYLEIGH_MAX_X:g}."
                )
                if _PMS_ERR is not None:
                    raise error from _PMS_ERR
                raise error
            warnings.warn(
                "PyMieScatt is unavailable; using the explicitly enabled "
                "Rayleigh-sphere approximation.",
                RuntimeWarning,
                stacklevel=2,
            )

        for rr, rh in enumerate(self.rh_grid):
            D_m = float(self.get_Dwet(RH=float(rh), T=self.temp, sigma_sa=self.get_surface_tension()))
            r_m = 0.5 * D_m
            area = math.pi * r_m * r_m  # geometric cross-section

            D_nm = D_m * 1e9
            for ww, lam_m in enumerate(self.wvl_grid):
                lam_nm = float(lam_m * 1e9)
                m = complex(self._mixture_ri(rr, ww))
                # out = MieQ(m, lam_nm, D_nm, asDict=True, asCrossSection=False)
                # # Convert efficiencies to absolute cross sections via geometric area
                # self.Cext[rr, ww] = out["Qext"] * area
                # self.Csca[rr, ww] = out["Qsca"] * area
                # self.Cabs[rr, ww] = out["Qabs"] * area

                if MieQ is not None:
                    out = MieQ(m, lam_nm, D_nm, asDict=True, asCrossSection=False)
                    # Convert efficiencies to absolute cross sections.
                    geom_area = np.pi / 4.0 * D_nm ** 2 * 1e-18 # from nm^2 to m^2
                    self.Cext[rr, ww] = out["Qext"] * geom_area
                    self.Csca[rr, ww] = out["Qsca"] * geom_area
                    self.Cabs[rr, ww] = out["Qabs"] * geom_area
                    self.g[rr, ww] = out["g"]
                else:
                    self._validate_rayleigh_domain(m, float(lam_m), r_m)
                    cext, csca, cabs, g = self._rayleigh_cross_sections(
                        m=m,
                        wavelength_m=float(lam_m),
                        radius_m=r_m,
                    )
                    self.Cext[rr, ww] = cext
                    self.Csca[rr, ww] = csca
                    self.Cabs[rr, ww] = cabs
                    self.g[rr, ww] = g


    # Convenience getters (unchanged)
    def get_cross_sections(self):
        return {
            "Cabs": self.Cabs,
            "Csca": self.Csca,
            "Cext": self.Cext,
            "g": self.g,
        }

    def get_refractive_indices(self):
        return {"dry_ri": self.dry_ris, "h2o_ri": self.h2o_ris}

    def get_cross_section(self, optics_type, rh_idx=None, wvl_idx=None):
        key = str(optics_type).lower()
        if key in ("b_abs", "absorption", "abs"):
            arr = self.Cabs
        elif key in ("b_scat", "scattering", "scat"):
            arr = self.Csca
        elif key in ("b_ext", "extinction", "ext"):
            arr = self.Cext
        elif key in ("g", "asymmetry"):
            arr = self.g
        else:
            raise ValueError(f"Unknown optics_type: {optics_type}")
        if rh_idx is not None and wvl_idx is not None:
            return arr[rh_idx, wvl_idx]
        return arr


def build(base_particle, config):
    """Optional fallback factory callable for discovery."""
    return HomogeneousParticle(base_particle, config)
