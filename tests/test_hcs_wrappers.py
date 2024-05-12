# for testing, run the following from the project folder:
#     pip install -e .
#     pytest --color=yes -v --cov=./src --cov-report=term-missing tests

import pytest
import anndata as ad
import json
import pandas as pd
import shutil
import zarr
import numpy as np
import dask.array
import matplotlib
from matplotlib import pyplot as plt
import warnings

from ez_zarr import hcs_wrappers

# fixtures --------------------------------------------------------------------
@pytest.fixture
def plate_3d():
    """A `hcs_wrappers.FractalZarr` object representing a 3D ome-zarr fileset"""
    return hcs_wrappers.FractalZarr('tests/example_data/plate_ones.zarr')

@pytest.fixture
def plate_2d():
    """A `hcs_wrappers.FractalZarr` object representing a 2D ome-zarr fileset"""
    return hcs_wrappers.FractalZarr('tests/example_data/plate_ones_mip.zarr', name="test")

@pytest.fixture
def plate_set1():
    """A `hcs_wrappers.FractalZarrSet` object representing a two ome-zarr filesets"""
    return hcs_wrappers.FractalZarrSet('tests/example_data')

@pytest.fixture
def plate_set2():
    """A `hcs_wrappers.FractalZarrSet` object representing a two ome-zarr filesets"""
    return hcs_wrappers.FractalZarrSet('tests/example_data', name="test")


# # ez_zarr.__main__ ------------------------------------------------------------
# def test_main():
#     result = subprocess.run(['python', '-m', 'ez_zarr', 'tests/example_data/plate_ones.zarr'],
#                             capture_output=True, text=True)
#     assert 'FractalZarr' in result.stdout
#     result = subprocess.run(['python', '-m', 'ez_zarr', 'tests/example_data/'],
#                             capture_output=True, text=True)
#     assert 'FractalZarrSet' in result.stdout
#     result = subprocess.run(['python', '-m', 'ez_zarr', 'does-not-exist'],
#                             capture_output=True, text=True)
#     assert 'usage' in result.stdout

# hcs_wrappers.FractalZarr ----------------------------------------------------
def test_digest_well_argument(plate_3d: hcs_wrappers.FractalZarr):
    """Test `FractalZarr._digest_well_argument`."""
    assert plate_3d._digest_well_argument(None) == 'B/03'
    assert plate_3d._digest_well_argument('B03') == 'B/03'
    assert plate_3d._digest_well_argument('B03', as_path=False) == 'B03'

def test_digest_include_wells_argument(plate_3d: hcs_wrappers.FractalZarr):
    """Test `FractalZarr._digest_include_wells_argument`."""
    assert plate_3d._digest_include_wells_argument([]) == ['B/03']
    assert plate_3d._digest_include_wells_argument(['B03']) == ['B/03']
    assert plate_3d._digest_include_wells_argument('B03') == ['B/03']

def test_digest_pyramid_level_argument(plate_3d: hcs_wrappers.FractalZarr, plate_2d: hcs_wrappers.FractalZarr):
    """Test `FractalZarr._digest_pyramid_level_argument`."""
    assert plate_3d._digest_pyramid_level_argument(None) == 2
    assert plate_3d._digest_pyramid_level_argument(1) == 1
    assert plate_3d._digest_pyramid_level_argument(1, ('image', '0')) == 1
    assert plate_2d._digest_pyramid_level_argument(1, ('label', 'organoids')) == 1
    assert plate_2d._digest_pyramid_level_argument(None, ('label', 'organoids')) == 2

def test_constructor_3d(plate_3d: hcs_wrappers.FractalZarr):
    """Test the `FractalZarr` constructor (3D)."""
    # exceptions
    with pytest.raises(Exception) as e_info:
        hcs_wrappers.FractalZarr('does-not-exist')
    with pytest.raises(Exception) as e_info:
        hcs_wrappers.FractalZarr('tests/example_data/plate_ones_mip.zarr/B')

    # expected values
    assert isinstance(plate_3d, hcs_wrappers.FractalZarr)
    assert plate_3d.name == 'plate_ones.zarr'
    assert isinstance(plate_3d._FractalZarr__top, zarr.Group)
    assert plate_3d.columns == [{'name': '03'}]
    assert plate_3d.rows == [{'name': 'B'}]
    assert len(plate_3d.wells) == 1
    assert len(plate_3d.channels) == 2
    assert isinstance(plate_3d.channels[0], dict)
    assert isinstance(plate_3d.multiscales_images, dict)
    assert 'datasets' in plate_3d.multiscales_images['0']
    assert len(plate_3d.multiscales_images['0']['datasets']) == 3
    assert plate_3d.level_paths_images['0'] == list(range(3))
    assert plate_3d.level_zyx_spacing_images['0'] == [[1.0, 0.1625, 0.1625], [1.0, 0.325, 0.325], [1.0, 0.65, 0.65]]
    assert list(plate_3d.level_zyx_scalefactor['0']) == [1., 2., 2.]
    assert plate_3d.label_names == []
    assert plate_3d.table_names == ['FOV_ROI_table']

def test_constructor_2d(plate_2d: hcs_wrappers.FractalZarr):
    """Test the `FractalZarr` constructor (2D)."""
    assert isinstance(plate_2d, hcs_wrappers.FractalZarr)
    assert plate_2d.name == 'test'
    assert isinstance(plate_2d._FractalZarr__top, zarr.Group)
    assert plate_2d.columns == [{'name': '03'}]
    assert plate_2d.rows == [{'name': 'B'}]
    assert len(plate_2d.wells) == 1
    assert len(plate_2d.channels) == 2
    assert isinstance(plate_2d.channels[0], dict)
    assert isinstance(plate_2d.multiscales_images, dict)
    assert 'datasets' in plate_2d.multiscales_images['0']
    assert len(plate_2d.multiscales_images['0']['datasets']) == 3
    assert plate_2d.level_paths_images['0'] == list(range(3))
    assert plate_2d.level_zyx_spacing_images['0'] == [[1.0, 0.1625, 0.1625], [1.0, 0.325, 0.325], [1.0, 0.65, 0.65]]
    assert list(plate_2d.level_zyx_scalefactor['0']) == [1., 2., 2.]
    assert plate_2d.label_names == ['organoids']
    assert plate_2d.table_names == ['FOV_ROI_table']

def test_plate_str(plate_2d: hcs_wrappers.FractalZarr, plate_3d: hcs_wrappers.FractalZarr):
    """Test `FractalZarr` object string representation."""
    assert str(plate_2d) == repr(plate_2d)
    assert str(plate_3d) == repr(plate_3d)

def test_missing_channel_attrs(tmpdir: str):
    """Test `FractalZarr` missing channel attributes."""
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
        hcs_wrappers.FractalZarr(str(tmpdir) + '/example_data/plate_ones_mip.zarr')

def test_missing_image_multiscales_attrs(tmpdir: str):
    """Test `FractalZarr` missing multiscales attributes."""
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
        hcs_wrappers.FractalZarr(str(tmpdir) + '/example_data/plate_ones_mip.zarr')

def test_missing_label_multiscales_attrs(tmpdir: str):
    """Test `FractalZarr` missing multiscales attributes."""
    # copy zarr fileset
    assert tmpdir.check()
    shutil.copytree('tests/example_data', str(tmpdir) + '/example_data')
    assert tmpdir.join('/example_data/plate_ones_mip.zarr/B/03/0/labels/organoids/.zattrs').check()
    # remove multiscales attributes from copy
    zattr_file = str(tmpdir) + '/example_data/plate_ones_mip.zarr/B/03/0/labels/organoids/.zattrs'
    with open(zattr_file) as f:
       zattr = json.load(f)
    del zattr['multiscales']
    with open(zattr_file, "w") as jsonfile:
        json.dump(zattr, jsonfile, indent=4)
    # test loading
    with pytest.raises(Exception) as e_info:
        hcs_wrappers.FractalZarr(str(tmpdir) + '/example_data/plate_ones_mip.zarr')

def test_missing_tables(tmpdir: str):
    """Test `FractalZarr` missing tables."""
    # copy zarr fileset
    assert tmpdir.check()
    shutil.copytree('tests/example_data', str(tmpdir) + '/example_data')
    assert tmpdir.join('/example_data/plate_ones_mip.zarr/B/03/0/.zattrs').check()
    # remove tables
    shutil.rmtree(str(tmpdir) + '/example_data/plate_ones_mip.zarr/B/03/0/tables')
    # test loading
    plate = hcs_wrappers.FractalZarr(str(tmpdir) + '/example_data/plate_ones_mip.zarr')
    assert plate.table_names == []

def test_get_path(plate_2d: hcs_wrappers.FractalZarr, plate_3d: hcs_wrappers.FractalZarr):
    """Test `FractalZarr.get_path().`"""
    assert plate_2d.get_path() == plate_2d.path
    assert plate_2d.get_path() == 'tests/example_data/plate_ones_mip.zarr'
    assert plate_3d.get_path() == plate_3d.path
    assert plate_3d.get_path() == 'tests/example_data/plate_ones.zarr'

def test_get_wells(plate_2d: hcs_wrappers.FractalZarr, plate_3d: hcs_wrappers.FractalZarr):
    """Test `FractalZarr.get_wells().`"""
    wells_expected = [{'columnIndex': 0, 'path': 'B/03', 'rowIndex': 0}]
    wells_expected_simple = ['B03']
    assert plate_2d.get_wells() == plate_2d.wells
    assert plate_2d.get_wells() == wells_expected
    assert plate_3d.get_wells() == plate_3d.wells
    assert plate_3d.get_wells() == wells_expected
    assert plate_2d.get_wells(simplify=True) == wells_expected_simple
    assert plate_3d.get_wells(simplify=True) == wells_expected_simple

def test_get_channels(plate_2d: hcs_wrappers.FractalZarr, plate_3d: hcs_wrappers.FractalZarr):
    """Test `FractalZarr.get_channels().`"""
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

def test_get_label_names(plate_2d: hcs_wrappers.FractalZarr, plate_3d: hcs_wrappers.FractalZarr):
    """Test `FractalZarr.get_label_names().`"""
    assert plate_2d.get_label_names() == ['organoids']
    assert plate_3d.get_label_names() == []

def test_get_table_names(plate_2d: hcs_wrappers.FractalZarr, plate_3d: hcs_wrappers.FractalZarr):
    """Test `FractalZarr.get_table_names().`"""
    assert plate_2d.get_table_names() == plate_2d.table_names
    assert plate_3d.get_table_names() == plate_3d.table_names

def test_get_table(plate_2d: hcs_wrappers.FractalZarr):
    """Test `FractalZarr.get_table`."""
    empty = plate_2d.get_table('does not exist')
    df = plate_2d.get_table('FOV_ROI_table')
    ann = plate_2d.get_table('FOV_ROI_table', as_AnnData=True)
    assert empty is None
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (4, 9)
    assert isinstance(ann, ad.AnnData)
    assert ann.shape == (4, 8)
    assert 'well' in ann.obs
    df2 = plate_2d.get_table('FOV_ROI_table', include_wells=['B03'], as_AnnData=False)
    assert df.equals(df2)

def test_get_image_ROI_3d(plate_3d: hcs_wrappers.FractalZarr):
    """Test `FractalZarr.get_image_ROI()`."""
    img0a = plate_3d.get_image_ROI(well=None, pyramid_level=2,
                                   upper_left_yx=None,
                                   lower_right_yx=None,
                                   size_yx=None,
                                   as_NumPy=False)
    img0b = plate_3d.get_image_ROI(well='B03', pyramid_level=2,
                                   upper_left_yx=(0, 0),
                                   lower_right_yx=(269, 319),
                                   size_yx=None,
                                   as_NumPy=True)
    assert isinstance(img0a, dask.array.Array)
    assert isinstance(img0b, np.ndarray)
    assert img0a.shape == (2, 3, 270, 320)
    assert (np.array(img0a) == img0b).all()
    
    with pytest.raises(Exception) as e_info:
        plate_3d.get_image_ROI(well='B03', pyramid_level=1,
                               upper_left_yx=(11, 10),
                               lower_right_yx=None, size_yx=None)

    img1a = plate_3d.get_image_ROI(well='B03', pyramid_level=1,
                                   pyramid_level_coord=0,
                                   upper_left_yx=(22, 20),
                                   lower_right_yx=(44, 40),
                                   size_yx=None,
                                   as_NumPy=True)
    img1b = plate_3d.get_image_ROI(well='B03', pyramid_level=1,
                                   upper_left_yx=(11, 10),
                                   lower_right_yx=None,
                                   size_yx=(11, 10),
                                   as_NumPy=True)
    img1c = plate_3d.get_image_ROI(well='B03', pyramid_level=1,
                                   upper_left_yx=None,
                                   lower_right_yx=(22, 20),
                                   size_yx=(11, 10),
                                   as_NumPy=True)
    assert isinstance(img1a, np.ndarray)
    assert isinstance(img1b, np.ndarray)
    assert isinstance(img1c, np.ndarray)
    assert img1a.shape == (2, 3, 12, 11)
    assert img1b.shape == img1a.shape
    assert img1c.shape == img1a.shape
    assert (img1b == img1a).all()
    assert (img1c == img1a).all()

def test_get_image_grid_ROIs_3d(plate_3d: hcs_wrappers.FractalZarr):
    """Test `FractalZarr.get_image_grid_ROIs().`"""
    # exceptions
    with pytest.raises(Exception) as e_info:
        plate_3d.get_image_grid_ROIs(num_x=2, num_y=2, num_select=5)
    
    with pytest.raises(Exception) as e_info:
        plate_3d.get_image_grid_ROIs(pyramid_level=2, sample_method='error')

    # sample_method = 'random'                        
    coord_1a, img_1a = plate_3d.get_image_grid_ROIs(well='B03', pyramid_level=2,
                                                    num_x=3, num_y=1, num_select=1,
                                                    sample_method='random', seed=1)
    coord_1b, img_1b = plate_3d.get_image_grid_ROIs(well='B03', pyramid_level=2,
                                                    num_x=3, num_y=1, num_select=1,
                                                    sample_method='random', seed=1)
    coord_2, img_2 = plate_3d.get_image_grid_ROIs(well='B03', pyramid_level=2,
                                                  num_x=3, num_y=1, num_select=1,
                                                  sample_method='random', seed=11)
    coord_3, img_3 = plate_3d.get_image_grid_ROIs(well='B03', pyramid_level=2,
                                                  num_x=8, num_y=8,
                                                  num_select=3,
                                                  sample_method='random', seed=3,
                                                  as_NumPy=True)
    assert len(coord_1a) == 1
    assert coord_1a == coord_1b
    assert len(coord_2) == 1
    assert coord_1a != coord_2
    assert all([isinstance(x, dask.array.Array) for x in img_1a])
    assert all(x.shape == (2, 3, 270, 106) for x in img_1a)
    assert all(x.shape == (2, 3, 270, 106) for x in img_1b)
    assert all(x.shape == (2, 3, 270, 106) for x in img_2)
    assert len(coord_3) == 3
    assert all([isinstance(x, np.ndarray) for x in img_3])
    assert all(x.shape == (2, 3, 33, 40) for x in img_3)

    # sample_method = 'sum'
    coord_4, img_4 = plate_3d.get_image_grid_ROIs(well='B03', pyramid_level=2,
                                                  num_x=3, num_y=1, num_select=3,
                                                  sample_method='sum')
    assert len(coord_4) == 3
    assert all([isinstance(x, dask.array.Array) for x in img_4])
    assert all(x.shape == (2, 3, 270, 106) for x in img_4)

    # sample_method = 'var'
    coord_5, img_5 = plate_3d.get_image_grid_ROIs(well='B03', pyramid_level=2,
                                                  num_x=3, num_y=1, num_select=3,
                                                  sample_method='var')
    assert len(coord_5) == 3
    assert all([isinstance(x, dask.array.Array) for x in img_5])
    assert all(x.shape == (2, 3, 270, 106) for x in img_5)

def test_get_image_table_idx(plate_2d: hcs_wrappers.FractalZarr):
    """Test `FractalZarr.get_image_table_idx()`."""
    with pytest.raises(Exception) as e_info:
        plate_2d.get_image_table_idx(table_name='does-not-exist', table_idx=0)
    img = plate_2d.get_image_table_idx(table_name='FOV_ROI_table',
                                       table_idx=0, well='B03', pyramid_level=2,
                                       as_NumPy=False)
    assert isinstance(img, dask.array.Array)
    assert img.shape == (2, 1, 136, 161)

def test_get_label_ROI_2d(plate_2d: hcs_wrappers.FractalZarr):
    """Test `FractalZarr.get_label_ROI()`."""
    msk0a = plate_2d.get_label_ROI(label_name='organoids',
                                   well=None, pyramid_level=2,
                                   upper_left_yx=None,
                                   lower_right_yx=None,
                                   size_yx=None,
                                   as_NumPy=False)
    msk0b = plate_2d.get_label_ROI(label_name='organoids',
                                   well='B03', pyramid_level=2,
                                   upper_left_yx=(0, 0),
                                   lower_right_yx=(269, 319),
                                   size_yx=None,
                                   as_NumPy=True)
    assert isinstance(msk0a, dask.array.Array)
    assert isinstance(msk0b, np.ndarray)
    assert msk0a.shape == (1, 270, 320)
    assert (np.array(msk0a) == msk0b).all()
    
    with pytest.raises(Exception) as e_info:
        plate_2d.get_label_ROI(label_name='organoids',
                               well='B03', pyramid_level=1,
                               upper_left_yx=(11, 10),
                               lower_right_yx=None, size_yx=None)

    with pytest.raises(Exception) as e_info:
        plate_2d.get_label_ROI(label_name='does-not-exist',
                               well='B03', pyramid_level=1,
                               upper_left_yx=(11, 10),
                               size_yx=(10, 10))

    msk1a = plate_2d.get_label_ROI(label_name='organoids',
                                   well='B03', pyramid_level=1,
                                   pyramid_level_coord=0,
                                   upper_left_yx=(22, 20),
                                   lower_right_yx=(44, 40),
                                   size_yx=None,
                                   as_NumPy=True)
    msk1b = plate_2d.get_label_ROI(label_name='organoids',
                                   well='B03', pyramid_level=1,
                                   upper_left_yx=(11, 10),
                                   lower_right_yx=None,
                                   size_yx=(11, 10),
                                   as_NumPy=True)
    msk1c = plate_2d.get_label_ROI(label_name='organoids',
                                   well='B03', pyramid_level=1,
                                   upper_left_yx=None,
                                   lower_right_yx=(22, 20),
                                   size_yx=(11, 10),
                                   as_NumPy=True)
    assert isinstance(msk1a, np.ndarray)
    assert isinstance(msk1b, np.ndarray)
    assert isinstance(msk1c, np.ndarray)
    assert msk1a.shape == (1, 12, 11)
    assert msk1b.shape == msk1a.shape
    assert msk1c.shape == msk1a.shape
    assert (msk1b == msk1a).all()
    assert (msk1c == msk1a).all()

def test_get_label_table_idx(plate_2d: hcs_wrappers.FractalZarr):
    """Test `FractalZarr.get_table_table_idx()`."""
    with pytest.raises(Exception) as e_info:
        plate_2d.get_label_table_idx(label_name='organoids', table_name='does-not-exist', table_idx=0)
    with pytest.raises(Exception) as e_info:
        plate_2d.get_label_table_idx(label_name='does-not-exist', table_name='FOV_ROI_table', table_idx=0)
    msk = plate_2d.get_label_table_idx(label_name='organoids', table_name='FOV_ROI_table',
                                       table_idx=0, well='B03', pyramid_level=2,
                                       as_NumPy=False)
    assert isinstance(msk, dask.array.Array)
    assert msk.shape == (1, 136, 161)

def test_coordinate_conversions(plate_3d: hcs_wrappers.FractalZarr, plate_2d: hcs_wrappers.FractalZarr):
    """Test `FractalZarr.convert_*` coordinate conversions."""
    # test exceptions
    with pytest.raises(Exception) as e_info:
        plate_3d.convert_micrometer_to_pixel('error')
    with pytest.raises(Exception) as e_info:
        plate_3d.convert_pixel_to_micrometer('error')
    with pytest.raises(Exception) as e_info:
        plate_3d.convert_pixel_to_pixel('error', 0, 0)
    with pytest.raises(Exception) as e_info:
        plate_2d.convert_micrometer_to_pixel(zyx=(1,2,3), pyramid_level=0,
                                             pyramid_ref=('label', 'does-not-exist'))

    # test expected results
    # ... image coordinates
    zyx_um_orig = (30, 40, 50)
    zyx_px_0 = plate_3d.convert_micrometer_to_pixel(zyx=zyx_um_orig, pyramid_level=0,
                                                    pyramid_ref=('image', '0'))
    zyx_px_1 = plate_3d.convert_micrometer_to_pixel(zyx=zyx_um_orig, pyramid_level=1)
    zyx_px_2 = plate_3d.convert_micrometer_to_pixel(zyx=zyx_um_orig, pyramid_level=2)
    zyx_um_from1 = plate_3d.convert_pixel_to_micrometer(zyx=zyx_px_1, pyramid_level=1)
    zyx_um_from2 = plate_3d.convert_pixel_to_micrometer(zyx=zyx_px_2, pyramid_level=2)
    zyx_px_0_from1 = plate_3d.convert_pixel_to_pixel(zyx=zyx_px_1, pyramid_level_from=1, pyramid_level_to=0)
    zyx_px_2_from1 = plate_3d.convert_pixel_to_pixel(zyx=zyx_px_1, pyramid_level_from=1, pyramid_level_to=2)
    assert isinstance(zyx_px_1, tuple)
    assert all([isinstance(v, np.int_) for v in zyx_px_1])
    assert len(zyx_px_1) == 3
    assert zyx_px_1 == (30, 123, 154)
    assert zyx_um_from1 == zyx_um_orig
    assert zyx_um_from2 == zyx_um_orig
    assert zyx_px_0_from1 == zyx_px_0
    assert zyx_px_2_from1 == zyx_px_2

    # ... label coordinates
    ref = ('label', 'organoids')
    zyx_px_0 = plate_2d.convert_micrometer_to_pixel(zyx=zyx_um_orig, pyramid_level=0,
                                                    pyramid_ref=ref)
    zyx_px_1 = plate_2d.convert_micrometer_to_pixel(zyx=zyx_um_orig, pyramid_level=1, pyramid_ref=ref)
    zyx_px_2 = plate_2d.convert_micrometer_to_pixel(zyx=zyx_um_orig, pyramid_level=2, pyramid_ref=ref)
    zyx_um_from1 = plate_2d.convert_pixel_to_micrometer(zyx=zyx_px_1, pyramid_level=1, pyramid_ref=ref)
    zyx_um_from2 = plate_2d.convert_pixel_to_micrometer(zyx=zyx_px_2, pyramid_level=2, pyramid_ref=ref)
    zyx_px_0_from1 = plate_2d.convert_pixel_to_pixel(zyx=zyx_px_1, pyramid_level_from=1, pyramid_level_to=0, 
                                                     pyramid_ref_from=ref, pyramid_ref_to=ref)
    zyx_px_2_from1 = plate_2d.convert_pixel_to_pixel(zyx=zyx_px_1, pyramid_level_from=1, pyramid_level_to=2, 
                                                     pyramid_ref_from=ref, pyramid_ref_to=ref)
    assert isinstance(zyx_px_1, tuple)
    assert all([isinstance(v, np.int_) for v in zyx_px_1])
    assert len(zyx_px_1) == 3
    assert zyx_px_1 == (30, 123, 154)
    assert zyx_um_from1 == zyx_um_orig
    assert zyx_um_from2 == zyx_um_orig
    assert zyx_px_0_from1 == zyx_px_0
    assert zyx_px_2_from1 == zyx_px_2

def test_calc_average_FOV(tmpdir: str, plate_3d: hcs_wrappers.FractalZarr):
    """Test `FractalZarr.calc_average_FOV().`"""
    # test exceptions
    # ... copy zarr fileset
    assert tmpdir.check()
    shutil.copytree('tests/example_data', str(tmpdir) + '/example_data')
    assert tmpdir.join('/example_data/plate_ones.zarr/B/03/0/tables/FOV_ROI_table').check()
    # ... remove tables
    shutil.rmtree(str(tmpdir) + '/example_data/plate_ones.zarr/B/03/0/tables')
    # ... test calculation
    plate_tmp = hcs_wrappers.FractalZarr(str(tmpdir) + '/example_data/plate_ones.zarr')
    with pytest.raises(Exception) as e_info:
        plate_tmp.calc_average_FOV()

    # test expected results
    avg0 = plate_3d.calc_average_FOV(pyramid_level=0)
    avg1 = plate_3d.calc_average_FOV(pyramid_level=1)
    avg2 = plate_3d.calc_average_FOV(pyramid_level=2)

    assert isinstance(avg0, np.ndarray)
    assert isinstance(avg1, np.ndarray)
    assert isinstance(avg2, np.ndarray)

    assert avg0.shape == (3, 540, 640)
    assert avg1.shape == (3, 270, 320)
    assert avg2.shape == (3, 135, 160)

def test_plot_plate(plate_2d: hcs_wrappers.FractalZarr, tmpdir: str):
    """Test plot_plate."""
    matplotlib.use('Agg')  # Use the 'Agg' backend, which doesn't need a display
    with warnings.catch_warnings():
        warnings.simplefilter('ignore') # suppress warning due to cannot show FigureCanvas 
        # without mask
        plate_2d.plot_plate(image_name='0',
                            label_name=None,
                            pyramid_level=None,
                            channels=[0],
                            channel_colors=['white'],
                            channel_ranges=[[0.01, 0.99]],
                            z_projection_method='maximum',
                            plate_layout='6well')
        # with mask
        plate_2d.plot_plate(image_name='0',
                            label_name='organoids',
                            pyramid_level=None,
                            channels=[0],
                            channel_colors=['white'],
                            channel_ranges=[[0.01, 0.99]],
                            z_projection_method='maximum',
                            plate_layout='6well')
        # fig_style='brightfield'
        plate_2d.plot_plate(image_name='0',
                            label_name='organoids',
                            pyramid_level=None,
                            fig_style='brightfield',
                            plate_layout='6well')

    plt.savefig(tmpdir.join('output.png'))
    assert True # check if the plotting ran through

def test_plot_well(plate_2d: hcs_wrappers.FractalZarr, tmpdir: str):
    """Test plot_plate."""
    matplotlib.use('Agg')  # Use the 'Agg' backend, which doesn't need a display
    with warnings.catch_warnings():
        warnings.simplefilter('ignore') # suppress warning due to cannot show FigureCanvas 
        plate_2d.plot_well(well='B03',
                           image_name='0',
                           label_name=None,
                           pyramid_level=None,
                           pyramid_level_coord=None,
                           channels=[0],
                           channel_colors=['white'],
                           channel_ranges=[[0.01, 0.99]],
                           z_projection_method='maximum',
                           scalebar_micrometer=50,
                           scalebar_label=True)
        plate_2d.plot_well(well='B03',
                           image_name='0',
                           label_name='organoids',
                           pyramid_level=None,
                           pyramid_level_coord=None,
                           channels=[0],
                           channel_colors=['white'],
                           channel_ranges=[[0.01, 0.99]],
                           z_projection_method='maximum')
    plt.savefig(tmpdir.join('output.png'))
    assert True # check if the plotting ran through

# hcs_wrappers.FractalZarrSet -------------------------------------------------
def test_constructor_set(plate_set1: hcs_wrappers.FractalZarrSet,
                         plate_set2: hcs_wrappers.FractalZarrSet,
                         tmpdir: str):
    """Test `FractalZarrSet` object constructor."""
    # exceptions
    with pytest.raises(Exception) as e_info:
        hcs_wrappers.FractalZarrSet('error')
    
    with pytest.raises(Exception) as e_info:
        hcs_wrappers.FractalZarrSet(tmpdir)

    # expected values    
    assert isinstance(plate_set1, hcs_wrappers.FractalZarrSet)
    assert plate_set1.path == 'tests/example_data'
    assert plate_set1.name == 'example_data'
    assert plate_set2.name == 'test'
    assert plate_set1.zarr_mip_idx == 0
    assert plate_set1.zarr_3d_idx == 1

def test_plateset_str(plate_set1: hcs_wrappers.FractalZarrSet):
    """Test `FractalZarrSet` object string representation."""
    assert str(plate_set1) == repr(plate_set1)

def test_plateset_get_element(plate_set1: hcs_wrappers.FractalZarrSet):
    """Test `FractFmiZarr.get_element()`."""
    assert len(plate_set1) == 2
    assert plate_set1[0] == plate_set1.zarr[0]
    assert plate_set1[plate_set1.zarr_names[1]] == plate_set1.zarr[1]

def test_plateset_get_plate_attributes(plate_set1: hcs_wrappers.FractalZarrSet):
    # exceptions
    with pytest.raises(Exception) as e_info:
        plate_set1.does_not_exist

    # expected values    
    assert plate_set1.name == 'example_data'
    assert plate_set1.wells == plate_set1.get_wells()
    assert plate_set1.get_table_names() == [['FOV_ROI_table'], ['FOV_ROI_table']]
