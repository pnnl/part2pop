
def patch_pymiescatt():
    # pymiescatt is outdated
    # TODO
    # see when this gets incorporated https://github.com/bsumlin/PyMieScatt/pull/26#issuecomment-2603303130
    import scipy.integrate as si
    trap = getattr(si, "trapz", None)
    trapezoid = getattr(si, "trapezoid", None)
    if trapezoid is None and trap is not None:
        si.trapezoid = trap
    elif trap is None and trapezoid is not None:
        si.trapz = trapezoid
