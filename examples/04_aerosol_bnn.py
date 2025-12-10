#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 14:13:57 2025

@author: beel083
"""

# %%

from part2pop.population import build_population
from part2pop.optics import build_optical_population
import numpy as np 
import warnings

# warnings.filterwarnings("ignore")

# lognormal size distribution parameters
Ntot = 1e9
GMD = 100e-9
GSD = 1.6
frac_BC = 0.2
N_bins = 5 

# Internally mixed: each particle has both BC and sulfate
pop_cfg_internallyMixed = {
    "type": "binned_lognormals",
    "N": [Ntot],
    "GMD": [GMD],
    "GSD": [GSD],
    "aero_spec_names": [["SO4","BC"]], # one population of internally mixed particles
    "aero_spec_fracs": [[(1.-frac_BC), frac_BC]], # mass fraction of each species in each particle
    "N_bins": N_bins,
    "N_sigmas": 5, # D_range is +/- 5 geometric standard deviations
  }

# %%

pop_configs = [pop_cfg_internallyMixed]
pop_int = build_population(pop_cfg_internallyMixed)


optics_config = {"type": "fractal", "method": "bnn", "Rmon": 15e-9, "wvl_grid": [550e-9], "rh_grid": [0.95]}
optical_pop_int = build_optical_population(pop_int, optics_config)

# print(optical_pop_int.Cabs)


# %%
'''
from part2pop.viz.style import StyleManager, Theme
from part2pop.viz.builder import build_plotter
mgr = StyleManager(Theme(), deterministic=False)

series = [
    {"key": "int", "population": pop_int, "label": "Fully internal mixture"},
]
line_styles = mgr.plan("line", [s["key"] for s in series])

import matplotlib.pyplot as plt

varname = "b_abs"
fig, ax = plt.subplots()
for s in series:
    cfg = {
        "varname": varname,
        "var_cfg": {"morphology":"homogeneous", "wvl_grid": np.linspace(350e-9,1050e-9,29), "rh_grid": [0.]},  # simple case: single x
        "style": line_styles[s["key"]],
    }
    plotter = build_plotter("state_line", cfg)
    plotter.plot(s["population"], ax, label=s["label"]+" (homogeneous)")
    
    cfg_cs = cfg.copy()
    cfg_cs["var_cfg"]["morphology"] = "core-shell"
    cfg_cs["style"] = line_styles[s["key"]].copy()
    cfg_cs["style"]["linestyle"] = "--"
    cfg_cs["style"]["linewidth"] = 2*cfg["style"]["linewidth"]
    plotter_cs = build_plotter("state_line", cfg_cs)
    plotter_cs.plot(s["population"], ax, label=s["label"]+" (core-shell)")
    
    cfg_frac = cfg.copy()
    cfg_frac["var_cfg"]["morphology"] = "fractal"
    cfg_frac["style"] = line_styles[s["key"]].copy()
    cfg_frac["style"]["linestyle"] = "dotted"
    cfg_frac["style"]["linewidth"] = 2*cfg["style"]["linewidth"]
    plotter_frac = build_plotter("state_line", cfg_frac)
    plotter_frac.plot(s["population"], ax, label=s["label"]+" (fractal)")

ax.legend(); fig.tight_layout()

plt.show()
'''