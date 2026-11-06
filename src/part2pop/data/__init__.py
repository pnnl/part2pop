import io
import os
from functools import lru_cache
from typing import Optional, TextIO, Tuple

import importlib_resources


def _resource(filename: str):
    """Locate ``filename`` relative to the packaged data directory.

    An absolute ``filename`` (for example ``specdata_path + '/aero_data.dat'``)
    resolves to that path itself, so user-supplied data files also pass
    through here.
    """
    return importlib_resources.files("part2pop.data").joinpath(filename)


def _file_stamp(resource) -> Optional[Tuple[int, int]]:
    """Return ``(mtime_ns, size)`` for a resource on the filesystem, else None."""
    if not isinstance(resource, os.PathLike):
        return None
    try:
        stat = os.stat(resource)
    except OSError:
        return None
    return (stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=32)
def _read_dataset(filename: str, encoding: str, stamp: Optional[Tuple[int, int]]) -> str:
    """Read a data file; ``stamp`` ties each cache entry to the file's state."""
    with _resource(filename).open("r", encoding=encoding) as f:
        return f.read()


def open_dataset(filename: str, encoding: str = "utf-8") -> TextIO:
    """Return a new text stream over the cached contents of a data file.

    ``filename`` is a packaged resource path such as
    ``"species_data/aero_data.dat"`` or an absolute path to a user data file.
    Up to 32 entries are cached, keyed by ``(filename, encoding)`` and, for
    files on the filesystem, by modification time and size, so an edited file
    is read again on the next call. Resources that are not plain files (for
    example inside a zip archive) are cached for the life of the process.

    Each call returns an independent ``io.StringIO`` with its own cursor. It is
    not an OS file handle: ``name``, ``fileno()`` and similar file attributes
    are not available.
    """
    stamp = _file_stamp(_resource(filename))
    return io.StringIO(_read_dataset(filename, encoding, stamp))
