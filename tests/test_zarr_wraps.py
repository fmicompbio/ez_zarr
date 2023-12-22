# for testing, run the following from the project folder:
#     pip install -e .
#     pytest
#     pytest --cov --color=yes --cov-report=term-missing

import anndata as ad
import json
import pandas as pd
import pytest
import shutil
import zarr

from easy_ome_zarr import zarr_wraps

# fixtures --------------------------------------------------------------------
@pytest.fixture
def plate_3d():
    return zarr_wraps.FmiZarr('tests/example_data/plate_ones.zarr')

@pytest.fixture
def plate_2d():
    return zarr_wraps.FmiZarr('tests/example_data/plate_ones_mip.zarr', name = "test")

@pytest.fixture
def plate_set():
    return zarr_wraps.FractalFmiZarr('tests/example_data')


# exeptions -------------------------------------------------------------------
def test_non_existing_path():
    with pytest.raises(Exception) as e_info:
        zarr_wraps.FmiZarr('does-not-exist')

def test_invalid_path():
    with pytest.raises(Exception) as e_info:
        zarr_wraps.FmiZarr('tests/example_data/plate_ones_mip.zarr/B')

# zarr_wraps.FmiZarr ----------------------------------------------------------
def test_constructor_3d(plate_3d):
    assert isinstance(plate_3d, zarr_wraps.FmiZarr)
    assert plate_3d.name == 'plate_ones.zarr'
    assert isinstance(plate_3d.top, zarr.Group)
    assert plate_3d.columns == [{'name': '03'}]
    assert plate_3d.rows == [{'name': 'B'}]
    assert len(plate_3d.wells) == 1
    assert len(plate_3d.channels) == 2
    assert isinstance(plate_3d.channels[0], dict)
    assert isinstance(plate_3d.multiscales, dict)
    assert 'datasets' in plate_3d.multiscales
    assert len(plate_3d.multiscales['datasets']) == 3
    assert plate_3d.level_paths == list(range(3))
    assert plate_3d.level_zyx_spacing == [[1.0, 0.1625, 0.1625], [1.0, 0.325, 0.325], [1.0, 0.65, 0.65]]
    assert list(plate_3d.level_zyx_scalefactor) == [1., 2., 2.]
    assert plate_3d.label_names == []
    assert plate_3d.table_names == ['FOV_ROI_table']

def test_constructor_2d(plate_2d):
    assert isinstance(plate_2d, zarr_wraps.FmiZarr)
    assert plate_2d.name == 'test'
    assert isinstance(plate_2d.top, zarr.Group)
    assert plate_2d.columns == [{'name': '03'}]
    assert plate_2d.rows == [{'name': 'B'}]
    assert len(plate_2d.wells) == 1
    assert len(plate_2d.channels) == 2
    assert isinstance(plate_2d.channels[0], dict)
    assert isinstance(plate_2d.multiscales, dict)
    assert 'datasets' in plate_2d.multiscales
    assert len(plate_2d.multiscales['datasets']) == 3
    assert plate_2d.level_paths == list(range(3))
    assert plate_2d.level_zyx_spacing == [[1.0, 0.1625, 0.1625], [1.0, 0.325, 0.325], [1.0, 0.65, 0.65]]
    assert list(plate_2d.level_zyx_scalefactor) == [1., 2., 2.]
    assert plate_2d.label_names == ['organoids']
    assert plate_2d.table_names == ['FOV_ROI_table']

def test_str(plate_2d, plate_3d):
    assert str(plate_2d) == repr(plate_2d)
    assert str(plate_3d) == repr(plate_3d)

def test_missing_channel_attrs(tmpdir):
    # copy zarr fileset
    assert tmpdir.check()
    shutil.copytree('tests/example_data', str(tmpdir) + '/example_data')
    assert tmpdir.join('/example_data/plate_ones_mip.zarr/B/03/0/.zattrs').check()
    # remove omero attributes from copy
    zattr_file = str(tmpdir) + '/example_data/plate_ones_mip.zarr/B/03/0/.zattrs'
    with open(zattr_file) as f:
       zattr = json.load(f)
    del zattr['omero']
    with open(zattr_file, "w") as jsonfile:
        json.dump(zattr, jsonfile, indent=4)
    # test loading
    with pytest.raises(Exception) as e_info:
        zarr_wraps.FmiZarr(str(tmpdir) + '/example_data/plate_ones_mip.zarr')

def test_missing_multiscales_attrs(tmpdir):
    # copy zarr fileset
    assert tmpdir.check()
    shutil.copytree('tests/example_data', str(tmpdir) + '/example_data')
    assert tmpdir.join('/example_data/plate_ones_mip.zarr/B/03/0/.zattrs').check()
    # remove multiscales attributes from copy
    zattr_file = str(tmpdir) + '/example_data/plate_ones_mip.zarr/B/03/0/.zattrs'
    with open(zattr_file) as f:
       zattr = json.load(f)
    del zattr['multiscales']
    with open(zattr_file, "w") as jsonfile:
        json.dump(zattr, jsonfile, indent=4)
    # test loading
    with pytest.raises(Exception) as e_info:
        zarr_wraps.FmiZarr(str(tmpdir) + '/example_data/plate_ones_mip.zarr')

def test_missing_tables(tmpdir):
    # copy zarr fileset
    assert tmpdir.check()
    shutil.copytree('tests/example_data', str(tmpdir) + '/example_data')
    assert tmpdir.join('/example_data/plate_ones_mip.zarr/B/03/0/.zattrs').check()
    # remove tables
    shutil.rmtree(str(tmpdir) + '/example_data/plate_ones_mip.zarr/B/03/0/tables')
    # test loading
    plate = zarr_wraps.FmiZarr(str(tmpdir) + '/example_data/plate_ones_mip.zarr')
    assert plate.table_names == []

def test_get_path(plate_2d, plate_3d):
    assert plate_2d.get_path() == plate_2d.path
    assert plate_2d.get_path() == 'tests/example_data/plate_ones_mip.zarr'
    assert plate_3d.get_path() == plate_3d.path
    assert plate_3d.get_path() == 'tests/example_data/plate_ones.zarr'

def test_get_wells(plate_2d, plate_3d):
    wells_expected = [{'columnIndex': 0, 'path': 'B/03', 'rowIndex': 0}]
    assert plate_2d.get_wells() == plate_2d.wells
    assert plate_2d.get_wells() == wells_expected
    assert plate_3d.get_wells() == plate_3d.wells
    assert plate_3d.get_wells() == wells_expected

def test_get_channels(plate_2d, plate_3d):
    channels_expected = [{'wavelength_id': 'A01_C01', 'label': 'some-label-1',
                          'window': {'min': '0', 'max': '10', 'start': '0', 'end': '10'},
                          'color': '00FFFF'},
                         {'wavelength_id': 'A01_C02', 'label': 'some-label-2',
                          'window': {'min': '0', 'max': '10', 'start': '0', 'end': '10'},
                          'color': '00FFFF'}]
    assert plate_2d.get_channels() == plate_2d.channels
    assert plate_2d.get_channels() == channels_expected
    assert plate_3d.get_channels() == plate_3d.channels
    assert plate_3d.get_channels() == channels_expected

def test_get_table_2d(plate_2d):
    empty = plate_2d.get_table('does not exist')
    df = plate_2d.get_table('FOV_ROI_table')
    ann = plate_2d.get_table('FOV_ROI_table', as_AnnData = True)
    assert empty is None
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (4, 8)
    assert isinstance(ann, ad.AnnData)
    assert ann.shape == (4, 8)

# zarr_wraps.FractalFmiZarr ---------------------------------------------------
def test_constructor_set(plate_set):
    assert isinstance(plate_set, zarr_wraps.FractalFmiZarr)
    assert plate_set.path == 'tests/example_data'
