# optics/factory/fractal.py
import numpy as np
import math, warnings
from scipy.optimize import fsolve
from .registry import register
from ..base import OpticalParticle
from ..refractive_index import build_refractive_index

try:
    import pyBCabs.retrieval as pbca
except:
    raise ImportError("pyBCabs package required for fractal morphology")

from part2pop._patch import patch_pymiescatt
patch_pymiescatt()
try:
    from PyMieScatt import MieQCoreShell
    _PMS_ERR = None
except Exception as e:
    MieQCoreShell = None
    _PMS_ERR = e


@register("fractal")
class FractalParticle(OpticalParticle):
    """
    Fractal particle morphology optical particle model with RH and wavelength dependence.

    Constructor expects (base_particle, config) to align with the factory builder.

    Optional config (read by OpticalParticle or here):
      - rh_grid, wvl_grid, temp (K), specdata_path, species_modifications
      - single_scatter_albedo (fallback SSA when PyMieScatt is unavailable; default: 0.9)
    """

    def __init__(self, base_particle, config):
        super().__init__(base_particle, config)

        # Refractive indices are attached at the population level by the
        # optics builder; the base class's _attach_refractive_indices is
        # guarded and will no-op if the species already have wavelength-aware
        # RIs. Keep the call to the base preparation intact.

        # User-tunable fallback SSA (only used if PyMieScatt is missing)
        self.single_scatter_albedo = float(config.get("single_scatter_albedo", 0.9))

        # Precompute geometry & per-wavelength dry/water RIs
        self._prepare_geometry_and_ris()

        # Do the optics
        self.compute_optics()

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
        """
        Compute cross-sections and asymmetry parameter per (RH, wavelength).
        Prefer PyMieScatt if available; otherwise use a size-parameter-based fallback.
        """
        vol_core = self.get_vol_core()
        vol_mon = (4.0/3.0)*np.pi*(20e-9)**3
        Npp = vol_core/vol_mon
        warnings.warn("Scattering by fractal particles is not yet implemented! Using core-shell Mie Theory values.", UserWarning)
        
        for rr, rh in enumerate(self.rh_grid):
            D_m = float(self.get_Dwet(RH=float(rh), T=self.temp, sigma_sa=self.get_surface_tension()))
            r_m = 0.5 * D_m
            area = math.pi * r_m * r_m  # geometric cross-section
            vol_tot = (4.0/3.0)*np.pi*r_m**3
            Vratio = vol_tot/self.get_vol_core()
            mass_h2o = 1000.0*(vol_tot-self.get_vol_tot()) # kg
            mass_BC = self.get_spec_mass("BC")[0]
            Mtot_Mbc = (self.get_mass_tot()+mass_h2o)/mass_BC
            if Npp > 20:
                core_Df, _ = self.get_Df(Npp, Vratio)
            else:
                core_Df = 1.78
            for ww, lam_m in enumerate(self.wvl_grid):
                phase_shift = self.get_phase_shift(Npp, core_Df, lam_m)
                mShell = np.imag(complex(self._shell_ri(rr, ww)))
                if phase_shift <= 1.0:
                    MAC = pbca.small_PSP(Mtot_Mbc, lam_m*1e9, mShell)
                else:
                    MAC = pbca.large_PSP(Mtot_Mbc, phase_shift, lam_m*1e9, mShell)
                self.Cabs[rr, ww] = MAC*1e3*mass_BC  
                
                # Fall back to PyMieScatt for scattering and extinction
                mCore = complex(self.core_ris[ww])
                mShell = complex(self._shell_ri(rr, ww))
                lam_nm = lam_m*1e9
                D_core_nm = self.get_Dcore() * 1e9
                D_shell_nm = D_m*1e9
                out = MieQCoreShell(
                    mCore, mShell, lam_nm, D_core_nm, D_shell_nm,
                    asDict=True, asCrossSection=False
                )
                self.Csca[rr, ww] = out["Qsca"] * area
                self.Cext[rr, ww] = self.Cabs[rr, ww] + self.Csca[rr, ww]
                self.g[rr, ww]    = out["g"]
            
            
    def get_x(self, Npp, Vratio, a1=1.0844906985904168, a2=-0.03072545646660544, a3=-0.8083509246658951, x0=0.46):
        x_final = x0*a1*Npp**a2
        x_core = (x0-x_final)*np.exp(a3*(Vratio-1))+x_final    
        x_coated = (x0-(1/3))*np.exp(a3*(Vratio-1))+(1/3)   
        return x_core, x_coated    
    
    def get_Df(self, Npp, Vratio, a=11.660434788081579, b=-41.3869911885898, c=-23.827027276610817, x0=0.46, Df0=1.8):
        sphere_x = 1.0/3.0
        sphere_Df = 3.0
        core_x, coated_x = self.get_x(Npp, Vratio)
        core_Df = a*(np.exp(b*(core_x - sphere_x)) - np.exp(b*(x0 - sphere_x)))+Df0
        coated_Df = (sphere_Df+a*np.exp(b*(x0-sphere_x))-Df0)*(np.exp(c*(coated_x-sphere_x))-np.exp(c*(x0-sphere_x)))+Df0
        return core_Df, coated_Df
    
    def meff_solver(self, phi):
        target=phi*((np.power(complex(1.95,0.79),2)-1)/(np.power(complex(1.95,0.79),2)+2))
        def meff(x):
            return [np.real(((np.power(complex(x[0],x[1]),2)-1)/(np.power(complex(x[0],x[1]),2)+2)))-np.real(target),
                    np.imag(((np.power(complex(x[0],x[1]),2)-1)/(np.power(complex(x[0],x[1]),2)+2)))-np.imag(target)]
        sol=fsolve(meff, [1.0,0])
        return sol
    
    def get_phase_shift(self, Npp, Df, wavelength, r_monomer=20e-9, kf=1.2):
        Rg=r_monomer*np.power(Npp/kf,1/Df)
        phi=kf*np.power((Df+2)/Df,-3/2)*np.power(r_monomer/Rg,3-Df)
        sol=self.meff_solver(phi)
        meff=complex(sol[0],sol[1])
        rho=2*((2*np.pi*Rg/wavelength))*abs(meff-1)
        return rho
    
    def _shell_ri(self, rr: int, ww: int) -> complex:
        v_h2o = self.h2o_vols[rr]
        v_dry = self.shell_dry_vol
        if (v_h2o + v_dry) <= 0.0:
            return complex(1.0, 0.0)
        return (self.h2o_ris[ww] * v_h2o + self.dry_shell_ris[ww] * v_dry) / (v_h2o + v_dry)

def build(base_particle, config):
    """Optional fallback factory callable for discovery."""
    return FractalParticle(base_particle, config)
