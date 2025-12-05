import importlib_resources

def open_dataset(filename: str, encoding: str = "utf-8"):
    """Open a data file from package resources."""
    return importlib_resources.files('pyparticle.data').joinpath(filename).open('r', encoding=encoding)
    # with importlib_resources.files('pyparticle.data').joinpath(filename).open('r', encoding=encoding) as f:
    #     return f