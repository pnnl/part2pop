from __future__ import annotations
import numpy as np

try:
    from PyMieScatt import MieQ  # core-shell not needed for homogeneous baseline
    HAS_PYMIESCATT = True
except Exception:
    MieQ = None
    HAS_PYMIESCATT = False


def _global_edges(cfg, N_bins):
    D_min = cfg.get("D_min", None)
    D_max = cfg.get("D_max", None)
    if (D_min is None) ^ (D_max is None):
        raise ValueError("Provide both D_min and D_max, or neither.")
    if D_min is None:
        return None
    if not (D_min > 0 and D_max > 0 and D_min < D_max):
        raise ValueError("Invalid global D_min/D_max.")
    edges = np.logspace(np.log10(D_min), np.log10(D_max), int(N_bins) + 1)
    mids = np.sqrt(edges[:-1] * edges[1:])
    return edges, mids


def _mode_edges(GMD, GSD, N_bins, N_sigmas=5.0):
    lo = np.exp(np.log(GMD) - 0.5 * float(N_sigmas) * np.log(GSD))
    hi = np.exp(np.log(GMD) + 0.5 * float(N_sigmas) * np.log(GSD))
    edges = np.logspace(np.log10(lo), np.log10(hi), int(N_bins) + 1)
    mids = np.sqrt(edges[:-1] * edges[1:])
    return edges, mids


def _as_list_or_repeat(x, n):
    if isinstance(x, (list, tuple, np.ndarray)):
        return list(x)
    return [x] * n


def _get_scalar_ri(mods, lam_m):
    # For tests, we prefer constant RI via n_550/k_550; allow spectral slopes if provided.
    n550 = float(mods.get("n_550", 1.45))
    k550 = float(mods.get("k_550", 0.0))
    a_n = float(mods.get("alpha_n", 0.0))
    a_k = float(mods.get("alpha_k", 0.0))
    lam0 = 550e-9
    # Use power-law slopes if provided (small effect)
    n = n550 * (lam_m / lam0) ** a_n
    k = k550 * (lam_m / lam0) ** a_k
    return complex(n, k)


def pymiescatt_adapter(pop_cfg: dict, var_cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reference PyMieScatt computation for b_scat/b_abs (m^-1) given a binned_lognormals pop_cfg
    and an optics var_cfg providing 'wvl_grid' (meters) and single RH (use RH=0.0 for baseline).
    Matches the package binning and weight definitions.
    """
    if not HAS_PYMIESCATT:
        raise RuntimeError("PyMieScatt not available")

    assert pop_cfg.get("type") == "binned_lognormals", "Adapter supports binned_lognormals"
    N_list = list(pop_cfg["N"])
    GMD_list = list(pop_cfg["GMD"])
    GSD_list = list(pop_cfg["GSD"])
    N_bins = pop_cfg.get("N_bins", 100)
    N_bins_list = _as_list_or_repeat(N_bins, len(GMD_list))
    N_sigmas = float(pop_cfg.get("N_sigmas", 5.0))

    # species_modifications → only the (single) absorbing/scattering species matters in baseline
    spec_mods = pop_cfg.get("species_modifications", {}) or {}

    wvl_grid_m = np.asarray(var_cfg["wvl_grid"], dtype=float)

    # Global edges?
    global_edges = None
    if pop_cfg.get("D_min") is not None and pop_cfg.get("D_max") is not None:
        nb = N_bins_list[0] if len(set(N_bins_list)) == 1 else N_bins_list[0]
        global_edges = _global_edges(pop_cfg, nb)

    # Aggregate (sum over modes)
    b_scat = np.zeros_like(wvl_grid_m, dtype=float)
    b_abs = np.zeros_like(wvl_grid_m, dtype=float)

    import scipy.stats

    for mode_i, (Ntot, GMD, GSD, nb) in enumerate(zip(N_list, GMD_list, GSD_list, N_bins_list)):
        if global_edges is not None:
            edges_m, mids_m = global_edges
        else:
            edges_m, mids_m = _mode_edges(GMD, GSD, nb, N_sigmas=N_sigmas)

        pdf = scipy.stats.norm(loc=np.log10(GMD), scale=np.log10(GSD))
        bin_width = np.log10(mids_m[1]) - np.log10(mids_m[0])
        weights = pdf.pdf(np.log10(mids_m)) * bin_width
        # per-bin number concentration (m^-3)
        weights = float(Ntot) * weights / weights.sum()

        D_nm = mids_m * 1e9
        geom_nm2_to_m2 = (np.pi / 4.0) * (D_nm ** 2) * 1e-18  # nm^2 → m^2

        for j, lam_m in enumerate(wvl_grid_m):
            # prefer species 'SO4' then any single key
            mods = spec_mods.get("SO4") if "SO4" in spec_mods else (next(iter(spec_mods.values())) if spec_mods else {})
            m = _get_scalar_ri(mods, lam_m)
            lam_nm = float(lam_m * 1e9)

            Qext = np.empty_like(D_nm, dtype=float)
            Qsca = np.empty_like(D_nm, dtype=float)
            Qabs = np.empty_like(D_nm, dtype=float)
            for k, d_nm in enumerate(D_nm):
                out = MieQ(m, lam_nm, float(d_nm), asDict=True, asCrossSection=False)
                Qext[k] = out["Qext"]
                Qsca[k] = out["Qsca"]
                Qabs[k] = out["Qabs"]

            Cext = Qext * geom_nm2_to_m2
            Csca = Qsca * geom_nm2_to_m2
            Cabs = Qabs * geom_nm2_to_m2

            b_scat[j] += float(np.sum(Csca * weights))
            b_abs[j] += float(np.sum(Cabs * weights))

    return wvl_grid_m, b_scat, b_abs


def _phi_from_mass_fracs_and_rho(names, mass_fracs, get_rho):
    """Compute volume fractions phi_i proportional to mass_frac_i / rho_i and normalized."""
    rhos = [float(get_rho(n)) for n in names]
    mf = [float(m) for m in mass_fracs]
    vols = [m / rho for m, rho in zip(mf, rhos)]
    s = sum(vols)
    if s <= 0:
        # avoid division by zero
        return [0.0 for _ in vols]
    return [v / s for v in vols]


def pymiescatt_adapter_core_shell(pop_cfg: dict, var_cfg: dict, core_specs=("BC",)) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reference PyMieScatt core–shell computation for b_scat/b_abs (m^-1) at RH=0.
    Matches binned_lognormals binning & weights used by the package and the
    package's core/shell diameter construction from species volume fractions.

    Assumptions for the baseline test:
      - single core species (default 'BC'), single shell species (e.g., 'SO4')
      - RH grid contains 0.0 only (we focus on dry optical sizes here)
    """
    if not HAS_PYMIESCATT:
        raise RuntimeError("PyMieScatt not available")

    # reuse helpers
    N_list = list(pop_cfg["N"])
    GMD_list = list(pop_cfg["GMD"])
    GSD_list = list(pop_cfg["GSD"])
    N_bins = pop_cfg.get("N_bins", 100)
    N_bins_list = _as_list_or_repeat(N_bins, len(GMD_list))
    N_sigmas = float(pop_cfg.get("N_sigmas", 5.0))
    spec_mods = pop_cfg.get("species_modifications", {}) or {}

    wvl_grid_m = np.asarray(var_cfg["wvl_grid"], dtype=float)

    # global edges?
    global_edges = None
    if pop_cfg.get("D_min") is not None and pop_cfg.get("D_max") is not None:
        nb = N_bins_list[0] if len(set(N_bins_list)) == 1 else N_bins_list[0]
        global_edges = _global_edges(pop_cfg, nb)

    # helper to get species density via registry
    from PyParticle.species.registry import get_species

    # Aggregate outputs
    b_scat = np.zeros_like(wvl_grid_m, dtype=float)
    b_abs = np.zeros_like(wvl_grid_m, dtype=float)

    import scipy.stats
    try:
        from PyMieScatt import MieQCoreShell
    except Exception:
        MieQCoreShell = None

    for mode_i, (Ntot, GMD, GSD, nb, names, mass_fracs) in enumerate(
        zip(N_list, GMD_list, GSD_list, N_bins_list, pop_cfg.get("aero_spec_names"), pop_cfg.get("aero_spec_fracs"))
    ):
        if global_edges is not None:
            edges_m, mids_m = global_edges
        else:
            edges_m, mids_m = _mode_edges(GMD, GSD, nb, N_sigmas=N_sigmas)

        pdf = scipy.stats.norm(loc=np.log10(GMD), scale=np.log10(GSD))
        bin_width = np.log10(mids_m[1]) - np.log10(mids_m[0])
        weights = pdf.pdf(np.log10(mids_m)) * bin_width
        weights = float(Ntot) * weights / weights.sum()

        # compute volume fractions from mass fractions and densities
        # names is a list like ['BC','SO4'] and mass_fracs are corresponding
        phi_list = _phi_from_mass_fracs_and_rho(names, mass_fracs, lambda n: get_species(n, **spec_mods.get(n, {})).density)

        # determine which indices are core vs shell for this mode
        core_idx = [i for i, n in enumerate(names) if n in core_specs]
        shell_idx = [i for i in range(len(names)) if i not in core_idx]

        # precompute densities dict for RI lookup
        # for RI, get species-specific mods from spec_mods

        for j, lam_m in enumerate(wvl_grid_m):
            lam_nm = float(lam_m * 1e9)

            # For each bin, compute D_core and D_shell
            D_nm = mids_m * 1e9
            geom_nm2_to_m2 = (np.pi / 4.0) * (D_nm ** 2) * 1e-18

            Qext = np.zeros_like(D_nm)
            Qsca = np.zeros_like(D_nm)
            Qabs = np.zeros_like(D_nm)

            for k, D_shell_m in enumerate(mids_m):
                V_total = (np.pi / 6.0) * (D_shell_m ** 3)
                # core volume fraction
                phi_core = sum(phi_list[i] for i in core_idx) if core_idx else 0.0
                V_core = phi_core * V_total
                if V_core <= 0.0:
                    D_core_m = 0.0
                else:
                    D_core_m = (6.0 / np.pi * V_core) ** (1.0 / 3.0)

                D_core_nm = D_core_m * 1e9
                D_shell_nm = D_shell_m * 1e9

                # Refractive indices: core (single or first core), shell mixture
                # choose mods for species
                # core RI
                if core_idx:
                    core_name = names[core_idx[0]]
                    core_mods = spec_mods.get(core_name, {})
                    m_core = _get_scalar_ri(core_mods, lam_m)
                else:
                    m_core = complex(1.0, 0.0)

                # shell mixture RI: volume-weighted among shell species
                if shell_idx:
                    # compute per-shell-spec volume fraction relative to shell
                    shell_phis = [phi_list[i] for i in shell_idx]
                    ssum = sum(shell_phis)
                    if ssum <= 0:
                        m_shell = complex(1.0, 0.0)
                    else:
                        n_sh = 0.0
                        k_sh = 0.0
                        for ii, pidx in enumerate(shell_idx):
                            name = names[pidx]
                            mods = spec_mods.get(name, {})
                            mm = _get_scalar_ri(mods, lam_m)
                            frac = shell_phis[ii] / ssum
                            n_sh += mm.real * frac
                            k_sh += mm.imag * frac
                        m_shell = complex(n_sh, k_sh)
                else:
                    m_shell = complex(1.0, 0.0)

                # compute Qs: if core diameter <=0, fallback to homogeneous Mie
                if D_core_nm <= 0.0 or MieQCoreShell is None:
                    # fallback homogeneous on shell RI
                    out = MieQ(m_shell, lam_nm, float(D_shell_nm), asDict=True, asCrossSection=False)
                    Qext_k = out["Qext"]
                    Qsca_k = out["Qsca"]
                    Qabs_k = out["Qabs"]
                else:
                    out = MieQCoreShell(m_core, m_shell, lam_nm, float(D_core_nm), float(D_shell_nm), asDict=True, asCrossSection=False)
                    Qext_k = out.get("Qext", 0.0)
                    Qsca_k = out.get("Qsca", 0.0)
                    Qabs_k = out.get("Qabs", 0.0)

                Qext[k] = Qext_k
                Qsca[k] = Qsca_k
                Qabs[k] = Qabs_k

            Csca = Qsca * geom_nm2_to_m2
            Cabs = Qabs * geom_nm2_to_m2

            b_scat[j] += float(np.sum(Csca * weights))
            b_abs[j] += float(np.sum(Cabs * weights))

    return wvl_grid_m, b_scat, b_abs
