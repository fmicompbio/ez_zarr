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
import numpy as np
import dask

from easy_ome_zarr import zarr_wraps

# fixtures --------------------------------------------------------------------
@pytest.fixture
def plate_3d():
    return zarr_wraps.FmiZarr('tests/example_data/plate_ones.zarr')

@pytest.fixture
def plate_2d():
    return zarr_wraps.FmiZarr('tests/example_data/plate_ones_mip.zarr', name = "test")

@pytest.fixture
def plate_set1():
    return zarr_wraps.FractalFmiZarr('tests/example_data')

@pytest.fixture
def plate_set2():
    return zarr_wraps.FractalFmiZarr('tests/example_data', name = "test")


# exeptions -------------------------------------------------------------------
def test_non_existing_path():
    with pytest.raises(Exception) as e_info:
        zarr_wraps.FmiZarr('does-not-exist')

def test_invalid_path():
    with pytest.raises(Exception) as e_info:
        zarr_wraps.FmiZarr('tests/example_data/plate_ones_mip.zarr/B')

# zarr_wraps.FmiZarr ----------------------------------------------------------
def test_digest_well_argument(plate_3d: zarr_wraps.FmiZarr):
    assert plate_3d._digest_well_argument(None) == 'B/03'
    assert plate_3d._digest_well_argument('B03') == 'B/03'

def test_digest_include_wells_argument(plate_3d: zarr_wraps.FmiZarr):
    assert plate_3d._digest_include_wells_argument(None) == ['B/03']
    assert plate_3d._digest_include_wells_argument(['B03']) == ['B/03']

def test_digest_pyramid_level_argument(plate_3d: zarr_wraps.FmiZarr):
    assert plate_3d._digest_pyramid_level_argument(None) == 2
    assert plate_3d._digest_pyramid_level_argument(1) == 1

def test_constructor_3d(plate_3d: zarr_wraps.FmiZarr):
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

def test_constructor_2d(plate_2d: zarr_wraps.FmiZarr):
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

def test_plate_str(plate_2d: zarr_wraps.FmiZarr, plate_3d: zarr_wraps.FmiZarr):
    assert str(plate_2d) == repr(plate_2d)
    assert str(plate_3d) == repr(plate_3d)

def test_missing_channel_attrs(tmpdir: str):
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

def test_missing_multiscales_attrs(tmpdir: str):
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

def test_missing_tables(tmpdir: str):
    # copy zarr fileset
    assert tmpdir.check()
    shutil.copytree('tests/example_data', str(tmpdir) + '/example_data')
    assert tmpdir.join('/example_data/plate_ones_mip.zarr/B/03/0/.zattrs').check()
    # remove tables
    shutil.rmtree(str(tmpdir) + '/example_data/plate_ones_mip.zarr/B/03/0/tables')
    # test loading
    plate = zarr_wraps.FmiZarr(str(tmpdir) + '/example_data/plate_ones_mip.zarr')
    assert plate.table_names == []

def test_get_path(plate_2d: zarr_wraps.FmiZarr, plate_3d: zarr_wraps.FmiZarr):
    assert plate_2d.get_path() == plate_2d.path
    assert plate_2d.get_path() == 'tests/example_data/plate_ones_mip.zarr'
    assert plate_3d.get_path() == plate_3d.path
    assert plate_3d.get_path() == 'tests/example_data/plate_ones.zarr'

def test_get_wells(plate_2d: zarr_wraps.FmiZarr, plate_3d: zarr_wraps.FmiZarr):
    wells_expected = [{'columnIndex': 0, 'path': 'B/03', 'rowIndex': 0}]
    wells_expected_simple = ['B03']
    assert plate_2d.get_wells() == plate_2d.wells
    assert plate_2d.get_wells() == wells_expected
    assert plate_3d.get_wells() == plate_3d.wells
    assert plate_3d.get_wells() == wells_expected
    assert plate_2d.get_wells(simplify=True) == wells_expected_simple
    assert plate_3d.get_wells(simplify=True) == wells_expected_simple

def test_get_channels(plate_2d: zarr_wraps.FmiZarr, plate_3d: zarr_wraps.FmiZarr):
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

def test_get_table_names(plate_2d: zarr_wraps.FmiZarr, plate_3d: zarr_wraps.FmiZarr):
    assert plate_2d.get_table_names() == plate_2d.table_names
    assert plate_3d.get_table_names() == plate_3d.table_names

def test_get_table_2d(plate_2d: zarr_wraps.FmiZarr):
    empty = plate_2d.get_table('does not exist')
    df = plate_2d.get_table('FOV_ROI_table')
    ann = plate_2d.get_table('FOV_ROI_table', as_AnnData = True)
    assert empty is None
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (4, 9)
    assert isinstance(ann, ad.AnnData)
    assert ann.shape == (4, 8)
    assert 'well' in ann.obs
    df2 = plate_2d.get_table('FOV_ROI_table', include_wells = ['B03'], as_AnnData = False)
    assert df.equals(df2)

def test_get_image_rect_3d(plate_3d: zarr_wraps.FmiZarr):
    img0a = plate_3d.get_image_rect(well = None, pyramid_level = 2,
                                    upper_left = None,
                                    lower_right = None,
                                    width_height = None,
                                    as_NumPy = False)
    img0b = plate_3d.get_image_rect(well = 'B03', pyramid_level = 2,
                                    upper_left = (0, 0),
                                    lower_right = (319, 269),
                                    width_height = None,
                                    as_NumPy = True)
    assert isinstance(img0a, dask.array.Array)
    assert isinstance(img0b, np.ndarray)
    assert img0a.shape == (2, 3, 270, 320)
    assert (np.array(img0a) == img0b).all()
    
    with pytest.raises(Exception) as e_info:
        plate_3d.get_image_rect(well = 'B03', pyramid_level = 1,
                                upper_left = (10, 11),
                                lower_right = None, width_height = None)

    img1a = plate_3d.get_image_rect(well = 'B03', pyramid_level = 1,
                                    upper_left = (10, 11),
                                    lower_right = (20, 22),
                                    width_height = None,
                                    as_NumPy = True)
    img1b = plate_3d.get_image_rect(well = 'B03', pyramid_level = 1,
                                    upper_left = (10, 11),
                                    lower_right = None,
                                    width_height = (10, 11),
                                    as_NumPy = True)
    img1c = plate_3d.get_image_rect(well = 'B03', pyramid_level = 1,
                                    upper_left = None,
                                    lower_right = (20, 22),
                                    width_height = (10, 11),
                                    as_NumPy = True)
    assert isinstance(img1a, np.ndarray)
    assert isinstance(img1b, np.ndarray)
    assert isinstance(img1c, np.ndarray)
    assert img1a.shape == (2, 3, 12, 11)
    assert img1b.shape == img1a.shape
    assert img1c.shape == img1a.shape
    assert (img1b == img1a).all()
    assert (img1c == img1a).all()

def test_get_image_sampled_rects_3d(plate_3d: zarr_wraps.FmiZarr):
    # exceptions
    with pytest.raises(Exception) as e_info:
        plate_3d.get_image_sampled_rects(num_x = 2, num_y = 2, num_select = 5)
    
    with pytest.raises(Exception) as e_info:
        plate_3d.get_image_sampled_rects(pyramid_level = 2, sample_method = 'error')

    # sample_method = 'random'                        
    coord_1a, img_1a = plate_3d.get_image_sampled_rects(well = 'B03', pyramid_level = 2,
                                                        num_select = 3,
                                                        sample_method = 'random', seed = 1)
    coord_1b, img_1b = plate_3d.get_image_sampled_rects(well = 'B03', pyramid_level = 2,
                                                        num_select = 3,
                                                        sample_method = 'random', seed = 1)
    coord_2, img_2 = plate_3d.get_image_sampled_rects(well = 'B03', pyramid_level = 2,
                                                      num_select = 3,
                                                      sample_method = 'random', seed = 2)
    coord_3, img_3 = plate_3d.get_image_sampled_rects(well = 'B03', pyramid_level = 2,
                                                      num_x = 8, num_y = 8,
                                                      num_select = 3,
                                                      sample_method = 'random', seed = 3,
                                                      as_NumPy = True)
    assert len(coord_1a) == 3
    assert coord_1a == coord_1b
    assert len(coord_2) == 3
    assert coord_1a != coord_2
    assert all([isinstance(x, dask.array.Array) for x in img_1a])
    assert all(x.shape == (2, 3, 27, 32) for x in img_1a)
    assert all(x.shape == (2, 3, 27, 32) for x in img_1b)
    assert all(x.shape == (2, 3, 27, 32) for x in img_2)
    assert len(coord_3) == 3
    assert all([isinstance(x, np.ndarray) for x in img_3])
    assert all(x.shape == (2, 3, 33, 40) for x in img_3)

    # sample_method = 'sum'
    coord_4, img_4 = plate_3d.get_image_sampled_rects(well = 'B03', pyramid_level = 2,
                                                      num_select = 3,
                                                      sample_method = 'sum')
    assert len(coord_4) == 3
    assert all([isinstance(x, dask.array.Array) for x in img_4])
    assert all(x.shape == (2, 3, 27, 32) for x in img_4)

    # sample_method = 'var'
    coord_5, img_5 = plate_3d.get_image_sampled_rects(well = 'B03', pyramid_level = 2,
                                                      num_select = 3,
                                                      sample_method = 'var')
    assert len(coord_5) == 3
    assert all([isinstance(x, dask.array.Array) for x in img_5])
    assert all(x.shape == (2, 3, 27, 32) for x in img_5)

def test_calc_average_FOV(tmpdir: str, plate_3d: zarr_wraps.FmiZarr):
    # test exceptions
    # ... copy zarr fileset
    assert tmpdir.check()
    shutil.copytree('tests/example_data', str(tmpdir) + '/example_data')
    assert tmpdir.join('/example_data/plate_ones.zarr/B/03/0/tables/FOV_ROI_table').check()
    # ... remove tables
    shutil.rmtree(str(tmpdir) + '/example_data/plate_ones.zarr/B/03/0/tables')
    # ... test calculation
    plate_tmp = zarr_wraps.FmiZarr(str(tmpdir) + '/example_data/plate_ones.zarr')
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

# zarr_wraps.FractalFmiZarr ---------------------------------------------------
def test_constructor_set(plate_set1: zarr_wraps.FractalFmiZarr,
                         plate_set2: zarr_wraps.FractalFmiZarr,
                         tmpdir: str):
    # exceptions
    with pytest.raises(Exception) as e_info:
        zarr_wraps.FractalFmiZarr('error')
    
    with pytest.raises(Exception) as e_info:
        zarr_wraps.FractalFmiZarr(tmpdir)

    # expected values    
    assert isinstance(plate_set1, zarr_wraps.FractalFmiZarr)
    assert plate_set1.path == 'tests/example_data'
    assert plate_set1.name == 'example_data'
    assert plate_set2.name == 'test'
    assert plate_set1.zarr_mip_idx == 0
    assert plate_set1.zarr_3d_idx == 1

def test_plateset_str(plate_set1: zarr_wraps.FractalFmiZarr):
    assert str(plate_set1) == repr(plate_set1)

def test_plateset_get_element(plate_set1: zarr_wraps.FractalFmiZarr):
    assert len(plate_set1) == 2
    assert plate_set1[0] == plate_set1.zarr[0]
    assert plate_set1[plate_set1.zarr_names[1]] == plate_set1.zarr[1]
