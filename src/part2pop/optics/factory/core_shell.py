# optics/factory/core_shell.py
from .registry import register
import numpy as np
from ..base import OpticalParticle
from ..utils import m_to_nm
from ..refractive_index import build_refractive_index
import math


from part2pop._patch import patch_pymiescatt
patch_pymiescatt()
try:
    from PyMieScatt import MieQCoreShell
    _PMS_ERR = None
except Exception as e:
    MieQCoreShell = None
    _PMS_ERR = e

@register("core_shell")
class CoreShellParticle(OpticalParticle):
    def __init__(self, base_particle, config):
        super().__init__(base_particle, config)
        if MieQCoreShell is None:
            raise ImportError(f"PyMieScatt required for morphology='core_shell': {_PMS_ERR}")

        # Refractive indices should be attached at population construction time
        # by the optics builder. The base class _attach_refractive_indices is
        # safe and will only attach if needed.
        
        # Precompute geometry and “dry” RIs per wavelength
        self._prepare_geometry_and_ris()
        self.compute_optics()
    
    # todo: move to base OpticalParticle? Or add to HomogeneousParticle?
    def _prepare_geometry_and_ris(self):
        # Core/shell dry volumes from Particle
        self.core_vol = float(self.get_vol_core())
        # If you want an explicit helper, you can add get_vol_dry_shell() to Particle;
        # for now compute from available calls:
        self.shell_dry_vol = float(self.get_vol_dry() - self.get_vol_core())

        # Water volumes vs RH
        Ddry = float(self.get_Ddry())
        self.h2o_vols = np.zeros(len(self.rh_grid))
        for rr, rh in enumerate(self.rh_grid):
            Dw = float(self.get_Dwet(RH=float(rh), T=self.temp))
            self.h2o_vols[rr] = (math.pi/6.0) * (Dw**3 - Ddry**3) if rh > 0.0 else 0.0

        # Refractive indices per wavelength
        Nw = len(self.wvl_grid)
        self.core_ris = np.zeros(Nw, dtype=complex)
        self.dry_shell_ris = np.zeros(Nw, dtype=complex)
        self.h2o_ris = np.zeros(Nw, dtype=complex)

        # Volume-weighted mixing (core & dry shell) using Particle-provided partitions
        vks = self.get_vks()
        core_idx = self.idx_core()
        shell_idx = self.idx_dry_shell()
        h2o_idx = self.idx_h2o()

        for ww in range(Nw):
            # Core effective RI
            if self.core_vol > 0.0 and len(core_idx) > 0:
                n_core = 0.0; k_core = 0.0
                for ii in core_idx:
                    f = float(vks[ii] / self.core_vol)
                    n_core += self.species[ii].refractive_index.real_ri_fun(self.wvl_grid[ww]) * f
                    k_core += self.species[ii].refractive_index.imag_ri_fun(self.wvl_grid[ww]) * f
                self.core_ris[ww] = complex(n_core, k_core)
            else:
                self.core_ris[ww] = complex(1.0, 0.0)
            
            # Dry shell effective RI
            if self.shell_dry_vol > 0.0 and len(shell_idx) > 0:
                n_sh = 0.0; k_sh = 0.0
                for ii in shell_idx:
                    f = float(vks[ii] / self.shell_dry_vol)
                    n_sh += self.species[ii].refractive_index.real_ri_fun(self.wvl_grid[ww]) * f
                    k_sh += self.species[ii].refractive_index.imag_ri_fun(self.wvl_grid[ww]) * f
                self.dry_shell_ris[ww] = complex(n_sh, k_sh)
            else:
                self.dry_shell_ris[ww] = complex(1.0, 0.0)
            
            # Water RI
            n_w = self.species[h2o_idx].refractive_index.real_ri_fun(self.wvl_grid[ww])
            k_w = self.species[h2o_idx].refractive_index.imag_ri_fun(self.wvl_grid[ww])
            self.h2o_ris[ww] = complex(n_w, k_w)
    
    def _shell_ri(self, rr: int, ww: int) -> complex:
        v_h2o = self.h2o_vols[rr]
        v_dry = self.shell_dry_vol
        if (v_h2o + v_dry) <= 0.0:
            return complex(1.0, 0.0)
        return (self.h2o_ris[ww] * v_h2o + self.dry_shell_ris[ww] * v_dry) / (v_h2o + v_dry)
    
    def compute_optics(self):
        # todo: leave options for other optical core-shell optical models
        # Try PyMieScatt first
        try:
            from PyMieScatt import MieQCoreShell
            use_pymie = True
        except ImportError:
            use_pymie = False

        for rr, rh in enumerate(self.rh_grid):
            D_shell_m = self.get_Dwet(RH=rh, T=self.temp, sigma_sa=self.get_surface_tension())
            r_m = 0.5 * D_shell_m
            area = np.pi * r_m * r_m

            if use_pymie:
                D_shell_nm = D_shell_m * 1e9
                D_core_nm = self.get_Dcore() * 1e9
                #D_core_nm = max(1.0, self.core_frac * D_shell_nm)
                for ww, lam_m in enumerate(self.wvl_grid):
                    lam_nm = float(lam_m * 1e9)
                    mCore = complex(self.core_ris[ww])
                    mShell = complex(self._shell_ri(rr, ww))
                    out = MieQCoreShell(
                        mCore, mShell, lam_nm, D_core_nm, D_shell_nm,
                        asDict=True, asCrossSection=False
                    )
                    self.Cext[rr, ww] = out["Qext"] * area
                    self.Csca[rr, ww] = out["Qsca"] * area
                    self.Cabs[rr, ww] = out["Qabs"] * area
                    self.g[rr, ww]    = out["g"]
            else:
                raise ImportError("PyMieScatt is required for core-shell optics calculations.")
            

def build(base_particle, config):
    """Optional fallback factory callable for discovery."""
    return CoreShellParticle(base_particle, config)
