from __future__ import annotations

import numpy as np

from ..base import ParticleVariable, VariableMeta
from .registry import register_particle_variable
from part2pop.optics.builder import build_optical_population


@register_particle_variable("SSA")
class SSA(ParticleVariable):
    meta = VariableMeta(
        name="SSA",
        description="particle single-scattering albedo",
        units="",
        axis_names=("particle",),
        default_cfg={
            "morphology": "core-shell",
            "RH": None,
            "T": 293.15,
            "wvl": 550e-9,
            "species_modifications": {},
        },
        aliases=("ssa", "single_scattering_albedo"),
        scale="linear",
        short_label="SSA",
        long_label="single-scattering albedo",
    )

    def _build_optical_cfg(self):
        cfg = self.cfg
        morph = cfg.get("morphology", "core-shell")
        if morph == "core-shell":
            morph = "core_shell"

        rh = cfg.get("RH", None)
        if rh is None:
            rh_grid = list(cfg.get("rh_grid", [0.0]))
        else:
            rh_grid = [float(rh)]

        wvl = cfg.get("wvl", None)
        if wvl is None:
            wvl_grid = list(cfg.get("wvl_grid", cfg.get("wvls", [550e-9])))
        else:
            wvl_grid = [float(wvl)]

        return {
            "rh_grid": rh_grid,
            "wvl_grid": wvl_grid,
            "type": morph,
            "temp": float(cfg.get("T", 293.15)),
            "species_modifications": cfg.get("species_modifications", {}),
        }

    def compute_one(self, population, part_id):
        values = self.compute_all(population)
        ids = list(population.ids)
        idx = ids.index(part_id)
        return float(values[idx])

    def compute_all(self, population):
        ocfg = self._build_optical_cfg()
        optical_pop = build_optical_population(population, ocfg)
        cext = np.asarray(optical_pop.Cext[:, 0, 0], dtype=float)
        csca = np.asarray(optical_pop.Csca[:, 0, 0], dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            ssa = np.where(cext > 0.0, csca / cext, 0.0)
        return ssa


def build(cfg=None):
    cfg = cfg or {}
    return SSA(cfg)
