OPTICS_TYPE_MAP = {
    "total_abs": "Cabs",
    "pure_bc_abs": "Cabs_bc",
    "clear_abs": "Cabs_clear",
    "total_scat": "Csca",
    "pure_bc_scat": "Csca_bc",
    "clear_scat": "Csca_clear",
    "total_ext": "Cext",
    "pure_bc_ext": "Cext_bc",
    "clear_ext": "Cext_clear",
    # Extend as needed
}


def get_cross_section_array_from_population(population, optics_type, idx_rh=None, idx_wvl=None):
    """
    Get the cross-section array from a population or model using a standardized mapping.
    """
    array_name = OPTICS_TYPE_MAP.get(optics_type)
    if array_name is None:
        raise ValueError(f"Unknown optics_type: {optics_type}")
    arr = getattr(population, array_name)
    if idx_rh is not None and idx_wvl is not None:
        return arr[:, idx_rh, idx_wvl]
    return arr

# Unit helpers for optics morphologies
import numpy as np

M_TO_NM = 1e9
MMINVERSE_TO_MINVERSE = 1e-6  # handy constant (mm^-1 -> m^-1)


def m_to_nm(x):
    """Convert meters to nanometers, returns numpy array-like result."""
    return np.asarray(x) * M_TO_NM
