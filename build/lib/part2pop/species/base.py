#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AerosolSpecies class definition.
@author: Laura Fierce
"""
from dataclasses import dataclass
from typing import Optional
# from ..data_old import species_open
from ..data import open_dataset

@dataclass
class AerosolSpecies:
    """AerosolSpecies: the definition of an aerosol species in terms of species-
    specific parameters (no state information)"""
    name: str
    density: Optional[float] = None
    kappa: Optional[float] = None
    molar_mass: Optional[float] = None
    surface_tension: float = 0.072

    def __post_init__(self):
        # Load defaults from file only if any key value is None
        if self.density is None or self.kappa is None or self.molar_mass is None:
            with open_dataset('species_data/aero_data.dat') as data_file:
                found = False
                for line in data_file:
                    if line.strip().startswith("#"):
                        continue
                    if line.upper().startswith(self.name.upper()):
                        name_in_file, density, ions_in_solution, molar_mass, kappa = line.split()
                        if self.density is None:
                            self.density = float(density)
                        if self.kappa is None:
                            self.kappa = float(kappa)
                        if self.molar_mass is None:
                            self.molar_mass = float(molar_mass.replace('d','e'))
                        found = True
                        break
            if not found:
                raise ValueError(f"Species data for '{self.name}' not found in data file.")
            if self.density is None or self.kappa is None or self.molar_mass is None:
                raise ValueError(f"Could not set all required fields for '{self.name}'.")