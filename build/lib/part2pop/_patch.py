
def patch_pymiescatt():
    # pymiescatt is outdated
    # TODO
    # see when this gets incorporated https://github.com/bsumlin/PyMieScatt/pull/26#issuecomment-2603303130
    import scipy.integrate as si
    si.trapz = si.trapezoid