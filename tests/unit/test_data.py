import io
import os

from part2pop import data


class _FakeResource:
    def __init__(self):
        self.opens = 0

    def joinpath(self, filename):
        assert filename == "sample.txt"
        return self

    def open(self, mode, encoding):
        assert mode == "r"
        assert encoding == "utf-8"
        self.opens += 1
        return io.StringIO("alpha\nbeta\n")


def test_open_dataset_caches_contents_but_returns_independent_streams(monkeypatch):
    resource = _FakeResource()
    monkeypatch.setattr(data.importlib_resources, "files", lambda _: resource)
    data._read_dataset.cache_clear()

    with data.open_dataset("sample.txt") as first:
        assert first.readline() == "alpha\n"

    with data.open_dataset("sample.txt") as second:
        assert second.read() == "alpha\nbeta\n"

    assert resource.opens == 1
    data._read_dataset.cache_clear()


def test_open_dataset_rereads_an_edited_user_file(tmp_path):
    """User data files (absolute specdata_path) must not be served stale."""
    data_file = tmp_path / "aero_data.dat"
    data_file.write_text("SO4 1800\n", encoding="utf-8")
    data._read_dataset.cache_clear()

    with data.open_dataset(str(data_file)) as first:
        assert first.read() == "SO4 1800\n"

    data_file.write_text("SO4 1234\n", encoding="utf-8")
    stat = data_file.stat()
    # Guarantee a new modification time even on coarse-grained filesystems.
    os.utime(data_file, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))

    with data.open_dataset(str(data_file)) as second:
        assert second.read() == "SO4 1234\n"
    data._read_dataset.cache_clear()


def test_get_species_sees_an_edited_specdata_path(tmp_path):
    """End-to-end: an edited user aero_data.dat changes the species returned."""
    from part2pop.species.registry import get_species

    packaged = data.open_dataset("species_data/aero_data.dat").read()
    data_file = tmp_path / "aero_data.dat"
    data_file.write_text(packaged, encoding="utf-8")
    data._read_dataset.cache_clear()
    assert get_species("SO4", str(tmp_path)).density == 1800.0

    data_file.write_text(
        packaged.replace("SO4            1800", "SO4            1234"),
        encoding="utf-8",
    )
    stat = data_file.stat()
    os.utime(data_file, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    assert get_species("SO4", str(tmp_path)).density == 1234.0
    data._read_dataset.cache_clear()
