import pytest
pytest.importorskip("netCDF4")
from part2pop.population.factory.partmc import get_ncfile


def test_custom_prefix_and_nonfirst_file(tmp_path):
    out = tmp_path / 'out'
    out.mkdir()
    (out / 'another_0001_00000002.nc').touch()
    wanted = out / 'custom-name_0001_00000001.nc'
    wanted.touch()
    assert get_ncfile(out, 1, 1) == wanted


def test_no_files_raises_useful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match='No PartMC file'):
        get_ncfile(tmp_path, 1, 1)


def test_ambiguous_matching_files_rejected(tmp_path):
    (tmp_path / 'foo_0001_00000001.nc').touch()
    (tmp_path / 'bar_0001_00000001.nc').touch()
    with pytest.raises(ValueError, match='Ambiguous'):
        get_ncfile(tmp_path, 1, 1)
