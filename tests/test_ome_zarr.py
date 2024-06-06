# for testing, run the following from the project folder:
#     pip install -e .
#     pytest --color=yes -v --cov=./src --cov-report=term-missing tests

import pytest
import shutil
import json
import copy
import zarr
import dask.array
import numpy as np
import warnings
import matplotlib
from matplotlib import pyplot as plt
import anndata as ad
import pandas as pd

from ez_zarr import ome_zarr

# fixtures --------------------------------------------------------------------
@pytest.fixture
def img3d():
    """A `ome_zarr.Image` object representing a 3D image"""
    return ome_zarr.Image('tests/example_data/plate_ones.zarr/B/03/0')

@pytest.fixture
def img2d():
    """A `ome_zarr.Image` object representing a 2D image"""
    return ome_zarr.Image('tests/example_data/plate_ones_mip.zarr/B/03/0', name="test")


# ome_zarr.Image ----------------------------------------------------
# ... helper functions ..............................................
def test_load_multiscale_info(img2d: ome_zarr.Image, tmpdir: str):
    """Test `Image._load_multiscale_info`."""

    # missing multiscales attributes
    # ... copy zarr fileset
    assert tmpdir.check()
    shutil.copytree('tests/example_data/plate_ones_mip.zarr/B/03/0',
                    str(tmpdir) + '/example_img')
    assert tmpdir.join('/example_img/.zattrs').check()
    # ... copy original .zattrs
    zattr_file = str(tmpdir) + '/example_img/.zattrs'
    shutil.copyfile(zattr_file, zattr_file + '.orig')
    # ... remove multiscales attributes from copy
    with open(zattr_file) as f:
       zattr = json.load(f)
    del zattr['multiscales']
    with open(zattr_file, "w") as jsonfile:
        json.dump(zattr, jsonfile, indent=4)
    # ... test loading
    with pytest.raises(Exception) as e_info:
        ome_zarr.Image(str(tmpdir) + '/example_img')
    # ... restore original .zattrs
    shutil.move(zattr_file + '.orig', zattr_file)

    # missing label multiscales attrs
    assert tmpdir.join('/example_img/labels/organoids/.zattrs').check()
    # ... copy original .zattrs
    zattr_file = str(tmpdir) + '/example_img/labels/organoids/.zattrs'
    shutil.copyfile(zattr_file, zattr_file + '.orig')
    # ... remove multiscales attributes from copy
    with open(zattr_file) as f:
       zattr = json.load(f)
    del zattr['multiscales']
    with open(zattr_file, "w") as jsonfile:
        json.dump(zattr, jsonfile, indent=4)
    # ... test loading
    with pytest.raises(Exception) as e_info:
        ome_zarr.Image(str(tmpdir) + '/example_img')
    # ... restore original .zattrs
    shutil.move(zattr_file + '.orig', zattr_file)

    # multiple multiscales dictionaries
    assert tmpdir.join('/example_img/.zattrs').check()
    # ... copy original .zattrs
    zattr_file = str(tmpdir) + '/example_img/.zattrs'
    shutil.copyfile(zattr_file, zattr_file + '.orig')
    # ... duplicate multiscales attributes in copy
    with open(zattr_file) as f:
       zattr = json.load(f)
    zattr['multiscales'].append(zattr['multiscales'][0])
    with open(zattr_file, "w") as jsonfile:
        json.dump(zattr, jsonfile, indent=4)
    # ... test loading
    with pytest.warns(UserWarning):
        ome_zarr.Image(str(tmpdir) + '/example_img')
    # ... restore original .zattrs
    shutil.move(zattr_file + '.orig', zattr_file)

    # missing axes in multiscales attrs
    assert tmpdir.join('/example_img/.zattrs').check()
    # ... copy original .zattrs
    zattr_file = str(tmpdir) + '/example_img/.zattrs'
    shutil.copyfile(zattr_file, zattr_file + '.orig')
    # ... remove axes multiscales attributes in copy
    with open(zattr_file) as f:
       zattr = json.load(f)
    del zattr['multiscales'][0]['axes']
    with open(zattr_file, "w") as jsonfile:
        json.dump(zattr, jsonfile, indent=4)
    # ... test loading
    with pytest.raises(Exception) as e_info:
        ome_zarr.Image(str(tmpdir) + '/example_img')
    # ... restore original .zattrs
    shutil.move(zattr_file + '.orig', zattr_file)

    # test loading
    # ... intensity image
    out_multiscales_image = img2d._load_multiscale_info(img2d.zarr_group, False)
    assert isinstance(out_multiscales_image, dict)
    assert 'axes' in out_multiscales_image
    assert 'datasets' in out_multiscales_image
    assert isinstance(out_multiscales_image['axes'], list)
    assert isinstance(out_multiscales_image['datasets'], list)
    # ... labels
    out_multiscales_labels = {x: img2d._load_multiscale_info(img2d.zarr_group.labels[x], False) for x in img2d.label_names}
    assert 'organoids' in out_multiscales_labels
    assert isinstance(out_multiscales_labels['organoids'], dict)
    assert 'axes' in out_multiscales_labels['organoids']
    assert 'datasets' in out_multiscales_labels['organoids']
    assert isinstance(out_multiscales_labels['organoids']['axes'], list)
    assert isinstance(out_multiscales_labels['organoids']['datasets'], list)

def test_load_axes_unit(img2d: ome_zarr.Image):
    """Test `Image._load_axes_unit`."""

    # missing unit
    multiscale_dict = copy.deepcopy(img2d.multiscales_image)
    del multiscale_dict['axes'][1]['unit']
    del multiscale_dict['axes'][2]['unit']
    del multiscale_dict['axes'][3]['unit']
    assert img2d._load_axes_unit(multiscale_dict) == 'pixel'

    # unsupported unit
    multiscale_dict = copy.deepcopy(img2d.multiscales_image)
    multiscale_dict['axes'][2]['unit'] = 'parsec'
    with pytest.raises(Exception) as e_info:
        img2d._load_axes_unit(multiscale_dict)

    # test loading
    # ... inensity image
    out_axes_unit_image = img2d._load_axes_unit(img2d.multiscales_image)
    assert out_axes_unit_image == 'micrometer'
    # ... labels
    out_axes_unit_labels = {x: img2d._load_axes_unit(img2d.multiscales_labels[x]) for x in img2d.label_names}
    assert 'organoids' in out_axes_unit_labels
    assert out_axes_unit_labels['organoids'] == 'micrometer'

def test_load_channel_info(img3d: ome_zarr.Image, img2d: ome_zarr.Image):
    """Test `Image._load_channel_info`."""

    # input with missing axes
    with pytest.raises(Exception) as e_info:
        img3d._load_channel_info({})
    
    # example input
    assert img2d._load_channel_info(img2d.multiscales_image) == 'czyx'
    assert img3d._load_channel_info(img3d.multiscales_image) == 'czyx'
    
    # synthetic input
    assert img2d._load_channel_info(
        {'axes': [{'name': 'day', 'type': 'time'},
                  {'name': 'c', 'type': 'channel'},
                  {'name': 'y', 'type': 'space'},
                  {'name': 'x', 'type': 'space'}]}) == 'tcyx'
    assert img2d._load_channel_info(
        {'axes': [{'name': 'c', 'type': 'channel'},
                  {'name': 'y', 'type': 'space'},
                  {'name': 'x', 'type': 'space'}]}) == 'cyx'
    assert img2d._load_channel_info(
        {'axes': [{'name': 'y', 'type': 'space'},
                  {'name': 'x', 'type': 'space'}]}) == 'yx'

def test_extract_scale_spacings(img2d: ome_zarr.Image):
    """Test `Image._extract_scale_spacings`."""
    
    # input with missing axes
    with pytest.raises(Exception) as e_info:
        img2d._extract_scale_spacings({})
    
    # example input
    assert img2d._extract_scale_spacings(img2d.multiscales_image['datasets'][0]) == [0, [1.0, 1.0, 0.1625, 0.1625]]
    assert img2d._extract_scale_spacings(img2d.multiscales_labels['organoids']['datasets'][1]) == [1, [1.0, 0.325, 0.325]]
    
    # synthetic input
    assert img2d._extract_scale_spacings(
        {'path': 11, 'coordinateTransformations': [{'scale': [1, 2.0, 3.25]}]}
    ) == [11, [1.0, 2.0, 3.25]]
    assert img2d._extract_scale_spacings(
        {'path': 7, 'coordinateTransformations': [{'scale': [1, 2, 0.5]}]}
    ) == [7, [1.0, 2.0, 0.5]]

def test_find_path_of_lowest_level(img2d: ome_zarr.Image):
    """Test `Image._find_path_of_lowest_resolution_level`."""
    
    # input with missing axes
    assert img2d._find_path_of_lowest_resolution_level({}) == None
    
    # example input
    assert img2d._find_path_of_lowest_resolution_level(img2d.multiscales_image['datasets']) == '2'
    assert img2d._find_path_of_lowest_resolution_level(img2d.multiscales_labels['organoids']['datasets']) == '2'
    
    # synthetic input
    assert img2d._find_path_of_lowest_resolution_level(
        [{'path': 0, 'coordinateTransformations': [{'scale': [0.2, 0.2]}]},
         {'path': 1, 'coordinateTransformations': [{'scale': [0.1, 0.4]}]}
        ]) == '1'
    assert img2d._find_path_of_lowest_resolution_level(
        [{'path': 0, 'coordinateTransformations': [{'scale': [0.1, 0.2]}]},
         {'path': 1, 'coordinateTransformations': [{'scale': [0.1, 0.1]}]}
        ]) == '0'

def test_find_path_of_highest_resolution_level(img2d: ome_zarr.Image):
    """Test `Image._find_path_of_highest_resolution_level`."""

    # input with missing axes
    assert img2d._find_path_of_highest_resolution_level({}) == None
    
    # example input
    assert img2d._find_path_of_highest_resolution_level(img2d.multiscales_image['datasets']) == '0'
    assert img2d._find_path_of_highest_resolution_level(img2d.multiscales_labels['organoids']['datasets']) == '0'
    
    # synthetic input
    assert img2d._find_path_of_highest_resolution_level(
        [{'path': 0, 'coordinateTransformations': [{'scale': [0.2, 0.2]}]},
         {'path': 1, 'coordinateTransformations': [{'scale': [0.1, 0.4]}]}
        ]) == '0'
    assert img2d._find_path_of_highest_resolution_level(
        [{'path': 0, 'coordinateTransformations': [{'scale': [0.1, 0.2]}]},
         {'path': 1, 'coordinateTransformations': [{'scale': [0.1, 0.1]}]}
        ]) == '1'

def test_digest_pyramid_level_argument(img2d: ome_zarr.Image, img3d: ome_zarr.Image):
    """Test `Image._digest_pyramid_level_argument`."""

    # invalid input
    with pytest.raises(Exception) as e_info:
        img2d._digest_pyramid_level_argument(6)
    with pytest.raises(Exception) as e_info:
        img2d._digest_pyramid_level_argument(None, 'error')

    # 2D image with labels    
    assert img2d._digest_pyramid_level_argument(None) == '2'
    assert img2d._digest_pyramid_level_argument(1, None) == '1'
    assert img2d._digest_pyramid_level_argument(1, 'organoids') == '1'
    assert img2d._digest_pyramid_level_argument('1', 'organoids') == '1'
    assert img2d._digest_pyramid_level_argument(None, 'organoids') == '2'

    # 3D image without labels
    assert img3d._digest_pyramid_level_argument(None) == '2'
    assert img3d._digest_pyramid_level_argument(1) == '1'
    assert img3d._digest_pyramid_level_argument('1', None) == '1'

def test_digest_channels_labels(img2d: ome_zarr.Image):
    """Test `Image._digest_channels_labels`."""

    # invalid input
    with pytest.raises(Exception) as e_info:
        img2d._digest_channels_labels(None)
    with pytest.raises(Exception) as e_info:
        img2d._digest_channels_labels(['error'])

    # expected results
    assert img2d._digest_channels_labels(['some-label-1']) == [0]
    assert img2d._digest_channels_labels(['some-label-2', 'some-label-1']) == [1, 0]

def test_digest_bounding_box(img2d: ome_zarr.Image):
    """Test `Image._digest_bounding_box`."""
    
    # invalid input
    with pytest.raises(Exception) as e_info: # unknown label
        img2d._digest_bounding_box(label_name='error')
    with pytest.raises(Exception) as e_info: # missing coordinates
        img2d._digest_bounding_box(upper_left_yx=(5, 10))
    with pytest.raises(Exception) as e_info: # wrong dimensions
        img2d._digest_bounding_box(upper_left_yx=(5, 10, 10), size_yx=(1, 1, 1))
    with pytest.raises(Exception) as e_info: # unknown pyramid level
        img2d._digest_bounding_box(upper_left_yx=(5, 10), lower_right_yx=(20, 50), pyramid_level=7)
    with pytest.raises(Exception) as e_info: # unknown coord pyramid level
        img2d._digest_bounding_box(upper_left_yx=(5, 10), lower_right_yx=(20, 50), pyramid_level_coord=7, coordinate_unit='pixel')
    with pytest.raises(Exception) as e_info: # unknown coordinate unit
        img2d._digest_bounding_box(upper_left_yx=(5, 10), lower_right_yx=(20, 50), coordinate_unit = 'error')

    # expected results
    # ... full array
    assert img2d._digest_bounding_box() == [(0, 0), (270, 320)]
    # ... corners from different inputs
    assert img2d._digest_bounding_box(upper_left_yx=(5, 10),
                                      lower_right_yx=(20, 50),
                                      coordinate_unit='pixel') == [(5, 10), (20, 50)]
    assert img2d._digest_bounding_box(upper_left_yx=(5, 10),
                                      size_yx=(15, 40),
                                      coordinate_unit='pixel') == [(5, 10), (20, 50)]
    assert img2d._digest_bounding_box(size_yx=(15, 40),
                                      lower_right_yx=(20, 50),
                                      coordinate_unit='pixel') == [(5, 10), (20, 50)]
    # ... different pyramid levels
    assert img2d._digest_bounding_box(upper_left_yx=(5, 10),
                                      lower_right_yx=(20, 50),
                                      pyramid_level=1,
                                      pyramid_level_coord=2,
                                      coordinate_unit='pixel') == [(10, 20), (40, 100)]
    # ... micrometer coordinates
    s = img2d.get_scale(pyramid_level=1)
    assert img2d._digest_bounding_box(upper_left_yx=(0, 0),
                                      lower_right_yx=(s[-2] * 1000, s[-1] * 1000),
                                      pyramid_level=1,
                                      coordinate_unit='micrometer') == [(0, 0), (1000, 1000)]
    # ... label array
    assert img2d._digest_bounding_box(label_name='organoids') == [(0, 0), (270, 320)]

def test_get_bounding_box_for_label_value(img2d: ome_zarr.Image):
    """Test `Image._get_bounding_box_for_label_value`."""

    # available label values
    assert img2d._get_bounding_box_for_label_value(label_value=0, label_name='organoids', label_pyramid_level='2') == ((0, 0, 0), (1, 270, 320))
    assert img2d._get_bounding_box_for_label_value(label_value=1, label_name='organoids', label_pyramid_level='0') == ((0, 200, 100), (1, 401, 301))
    assert img2d._get_bounding_box_for_label_value(label_value=2, label_name='organoids', label_pyramid_level='0') == ((0, 600, 0), (1, 1001, 401))
    assert img2d._get_bounding_box_for_label_value(label_value=3, label_name='organoids', label_pyramid_level='0') == ((0, 400, 400), (1, 1001, 1001))
    assert img2d._get_bounding_box_for_label_value(label_value=4, label_name='organoids', label_pyramid_level='0') == (None, None)

def test_image_str(img2d: ome_zarr.Image, img3d: ome_zarr.Image):
    """Test `ome_zarr.Image` object string representation."""
    assert str(img2d) == repr(img2d)
    assert str(img3d) == repr(img3d)

# ... constructor ...................................................
def test_constructor(img2d: ome_zarr.Image, img3d: ome_zarr.Image, tmpdir: str):
    """Test `ome_zarr.Image` object constructor."""

    # non-conforming input
    # ... multiple zarr groups
    assert tmpdir.check()
    shutil.copytree('tests/example_data/plate_ones_mip.zarr/B/03/0',
                    str(tmpdir) + '/example_img')
    shutil.copytree('tests/example_data/plate_ones_mip.zarr/B/03/0/labels',
                    str(tmpdir) + '/example_img/labels_copy')
    with pytest.raises(Exception) as e_info:
        ome_zarr.Image(str(tmpdir) + '/example_img')
    shutil.rmtree(str(tmpdir) + '/example_img/labels_copy')
    # ... path without array
    with pytest.raises(Exception) as e_info:
        ome_zarr.Image('tests/example_data/')
    # ... missing omero
    zattr_file = str(tmpdir) + '/example_img/.zattrs'
    shutil.copyfile(zattr_file, zattr_file + '.orig')
    with open(zattr_file) as f:
       zattr = json.load(f)
    del zattr['omero']
    with open(zattr_file, "w") as jsonfile:
        json.dump(zattr, jsonfile, indent=4)
    img_no_omero = ome_zarr.Image(str(tmpdir) + '/example_img')
    assert img_no_omero.channels == [{'label': 'channel-1', 'color': '00FFFF'}, {'label': 'channel-2', 'color': '00FFFF'}]
    shutil.move(zattr_file + '.orig', zattr_file)
    # ... clean up
    shutil.rmtree(str(tmpdir) + '/example_img')

    # 2D image with labels
    assert isinstance(img2d, ome_zarr.Image)
    assert img2d.path == 'tests/example_data/plate_ones_mip.zarr/B/03/0'
    assert img2d.name == 'test'
    assert isinstance(img2d.zarr_group, zarr.Group)
    assert all([isinstance(x, zarr.Array) for x in img2d.array_dict.values()])
    assert img2d.ndim == 4
    assert img2d.label_names == ['organoids']
    assert img2d.table_names == ['FOV_ROI_table']
    assert img2d.axes_unit_image == 'micrometer'
    assert img2d.channel_info_image == 'czyx'
    assert img2d.channel_info_labels == {'organoids': 'zyx'}
    assert img2d.nchannels_image == 2
    assert isinstance(img2d.channels, list)
    assert all([isinstance(img2d.channels[i], dict) for i in range(img2d.nchannels_image)])

    # 3D image without labels
    assert isinstance(img3d, ome_zarr.Image)
    assert img3d.path == 'tests/example_data/plate_ones.zarr/B/03/0'
    assert img3d.name == '0'
    assert isinstance(img3d.zarr_group, zarr.Group)
    assert all([isinstance(x, zarr.Array) for x in img3d.array_dict.values()])
    assert img3d.ndim == 4
    assert img3d.label_names == []
    assert img3d.table_names == ['FOV_ROI_table']
    assert img3d.axes_unit_image == 'micrometer'
    assert img3d.channel_info_image == 'czyx'
    assert img3d.channel_info_labels == {}
    assert img3d.nchannels_image == 2
    assert isinstance(img3d.channels, list)
    assert all([isinstance(img3d.channels[i], dict) for i in range(img3d.nchannels_image)])

# ... accesssors ....................................................
def test_get_path(img2d: ome_zarr.Image):
    """Test `get_path()` method of `ome_zarr.Image` object."""
    assert img2d.get_path() == 'tests/example_data/plate_ones_mip.zarr/B/03/0'
    assert img2d.get_path() == img2d.path

def test_get_channels(img3d: ome_zarr.Image):
    """Test `get_channels()` method of `ome_zarr.Image` object."""
    assert isinstance(img3d.get_channels(), list)
    assert all([isinstance(x, dict) for x in img3d.get_channels()])
    assert all(['label' in x for x in img3d.get_channels()])

def test_get_label_names(img2d: ome_zarr.Image):
    """Test `get_label_names()` method of `ome_zarr.Image` object."""
    assert img2d.get_label_names() == ['organoids']
    assert img2d.get_label_names() == img2d.label_names

def test_get_table_names(img3d: ome_zarr.Image):
    """Test `get_table_names()` method of `ome_zarr.Image` object."""
    assert img3d.get_table_names() == ['FOV_ROI_table']
    assert img3d.get_table_names() == img3d.table_names

def test_get_pyramid_levels(img2d: ome_zarr.Image, img3d: ome_zarr.Image):
    """Test `get_pyramid_levels()` method of `ome_zarr.Image` object."""
    assert img2d.get_pyramid_levels() == ['0', '1', '2']
    assert img2d.get_pyramid_levels(None) == ['0', '1', '2']
    assert img2d.get_pyramid_levels('organoids') == ['0', '1', '2']
    assert img3d.get_pyramid_levels() == ['0', '1', '2']

def test_get_scale(img2d: ome_zarr.Image, img3d: ome_zarr.Image):
    """Test `ome_zarr.Image` object get_scale() method."""

    # invalid input
    with pytest.raises(Exception) as e_info:
        img2d.get_scale()
    with pytest.raises(Exception) as e_info:
        img2d.get_scale(7)
    with pytest.raises(Exception) as e_info:
        img2d.get_scale(1, 'error')

    # 2D image with labels
    assert img2d.get_scale(0) == [1.0, 1.0, 0.1625, 0.1625]
    assert img2d.get_scale('1') == [1.0, 1.0, 0.325, 0.325]
    assert img2d.get_scale(2) == [1.0, 1.0, 0.65, 0.65]
    assert img2d.get_scale(0, 'organoids') == [1.0, 0.1625, 0.1625]
    assert img2d.get_scale('1', 'organoids') == [1.0, 0.325, 0.325]
    assert img2d.get_scale(2, 'organoids') == [1.0, 0.65, 0.65]
    assert img2d.get_scale(2, 'organoids', True) == [1.0, 0.65, 0.65]

    # 3D image
    assert img3d.get_scale(0) == [1.0, 1.0, 0.1625, 0.1625]
    assert img3d.get_scale('1') == [1.0, 1.0, 0.325, 0.325]
    assert img3d.get_scale(2) == [1.0, 1.0, 0.65, 0.65]
    assert img3d.get_scale(2, spatial_axes_only=True) == [1.0, 0.65, 0.65]

def test_get_array_by_coordinate(img2d: ome_zarr.Image, img3d: ome_zarr.Image):
    """Test `ome_zarr.Image` object get_array_by_coordinate() method."""

    # invalid input
    with pytest.raises(Exception) as e_info: # unknown label
        img2d.get_array_by_coordinate('error')
    with pytest.raises(Exception) as e_info: # missing coordinates
        img2d.get_array_by_coordinate(upper_left_yx=(5, 10))
    with pytest.raises(Exception) as e_info: # wrong dimensions
        img2d.get_array_by_coordinate(upper_left_yx=(5, 10, 10), size_yx=(1, 1, 1))
    with pytest.raises(Exception) as e_info: # unknown pyramid level
        img2d.get_array_by_coordinate(upper_left_yx=(5, 10), lower_right_yx=(20, 50), pyramid_level = 7)
    with pytest.raises(Exception) as e_info: # unknown coord pyramid level
        img2d.get_array_by_coordinate(upper_left_yx=(5, 10), lower_right_yx=(20, 50), pyramid_level_coord = 7, coordinate_unit = 'pixel')
    with pytest.raises(Exception) as e_info: # unknown coordinate unit
        img2d.get_array_by_coordinate(upper_left_yx=(5, 10), lower_right_yx=(20, 50), coordinate_unit = 'error')

    # 2D image
    assert img2d.get_array_by_coordinate() == img2d.array_dict['2']
    img0a = img2d.get_array_by_coordinate(
        upper_left_yx=None,
        lower_right_yx=None,
        size_yx=None,
        pyramid_level=2,
        as_NumPy=False)
    img0b = img2d.get_array_by_coordinate(
        upper_left_yx=(0, 0),
        lower_right_yx=(269, 319),
        size_yx=None,
        pyramid_level=2,
        as_NumPy=True)
    assert isinstance(img0a, zarr.Array)
    assert isinstance(img0b, np.ndarray)
    assert img0a.shape == (2, 1, 270, 320)
    assert (np.array(img0a) == img0b).all()
    
    img1a = img3d.get_array_by_coordinate(
        upper_left_yx=(22, 20),
        lower_right_yx=(44, 40),
        size_yx=None,
        coordinate_unit='pixel',
        pyramid_level=1,
        pyramid_level_coord=0,
        as_NumPy=True)
    img1b = img3d.get_array_by_coordinate(
        upper_left_yx=(11, 10),
        lower_right_yx=None,
        size_yx=(11, 10),
        coordinate_unit='pixel',
        pyramid_level=1,
        as_NumPy=True)
    img1c = img3d.get_array_by_coordinate(
        upper_left_yx=None,
        lower_right_yx=(22, 20),
        size_yx=(11, 10),
        coordinate_unit='pixel',
        pyramid_level=1,
        as_NumPy=True)
    assert isinstance(img1a, np.ndarray)
    assert isinstance(img1b, np.ndarray)
    assert isinstance(img1c, np.ndarray)
    assert img1a.shape == (2, 3, 11, 10)
    assert img1b.shape == img1a.shape
    assert img1c.shape == img1a.shape
    assert (img1b == img1a).all()
    assert (img1c == img1a).all()

    img2a = img2d.get_array_by_coordinate(
        label_name='organoids',
        upper_left_yx=(0, 0),
        lower_right_yx=(11, 10),
        size_yx=None,
        coordinate_unit='micrometer',
        pyramid_level=0,
        as_NumPy=True)
    img2b = img2d.get_array_by_coordinate(
        label_name='organoids',
        upper_left_yx=None,
        lower_right_yx=(11, 10),
        size_yx=(11, 10),
        coordinate_unit='micrometer',
        pyramid_level=0,
        as_NumPy=True)
    assert isinstance(img2a, np.ndarray)
    assert isinstance(img2b, np.ndarray)
    expected_shape = np.divide(
        [1, 11, 10], # size z, y, x in micrometer
        img2d.get_scale(0, 'organoids'))
    expected_shape = tuple([int(round(x)) for x in expected_shape])
    assert img2a.shape == expected_shape
    assert img2b.shape == img2a.shape
    assert (img1b == img1a).all()
    assert (img1c == img1a).all()

def test_get_array_pair_by_coordinate(img2d: ome_zarr.Image, tmpdir: str):
    """Test `ome_zarr.Image` object get_array_pair_by_coordinate() method."""

    # using pyramid_level corresponding to a lower resolution intensity image
    #     than any available label resolutions
    # ... copy zarr fileset
    assert tmpdir.check()
    shutil.copytree('tests/example_data/plate_ones_mip.zarr/B/03/0',
                    str(tmpdir) + '/example_img')
    assert tmpdir.join('/example_img/1').check()
    # ... remove pyramid level 2 from label organoids
    shutil.rmtree(str(tmpdir) + '/example_img/labels/organoids/2')
    zattr_file = str(tmpdir) + '/example_img/labels/organoids/.zattrs'
    with open(zattr_file) as f:
        zattr = json.load(f)
    zattr['multiscales'][0]['datasets'] = zattr['multiscales'][0]['datasets'][:2]
    with open(zattr_file, "w") as jsonfile:
        json.dump(zattr, jsonfile, indent=4)
    # ... plot
    imgtmp = ome_zarr.Image(str(tmpdir) + '/example_img')
    with pytest.raises(Exception) as e_info:
        imgtmp.get_array_pair_by_coordinate(pyramid_level='2', label_name='organoids')
    # ... clean up
    shutil.rmtree(str(tmpdir) + '/example_img')

def test_get_table(img2d: ome_zarr.Image):
    """Test `Image.get_table()`."""

    # invalid input
    with pytest.warns(UserWarning):
        assert img2d.get_table('error') is None

    # table as AnnData
    res1 = img2d.get_table(table_name='FOV_ROI_table',
                           as_AnnData=True)
    assert isinstance(res1, ad.AnnData)
    assert res1.shape == (4, 8)

    # table as pandas.DataFrame
    res2 = img2d.get_table(table_name='FOV_ROI_table',
                           as_AnnData=False)
    assert isinstance(res2, pd.DataFrame)
    assert res2.shape == (4, 8)

def test_plot(img2d: ome_zarr.Image, tmpdir: str):
    """Test `Image.plot()`."""

    matplotlib.use('Agg')  # Use the 'Agg' backend, which doesn't need a display
    with warnings.catch_warnings():
        # suppress warning due to cannot show FigureCanvas
        warnings.simplefilter('ignore')

        # using channels_labels
        img2d.plot(label_name=None,
                   pyramid_level=None,
                   pyramid_level_coord=None,
                   channels_labels=['some-label-1'],
                   channel_colors=['white'],
                   channel_ranges=[[0.01, 0.99]],
                   z_projection_method='maximum',
                   scalebar_micrometer=50,
                   show_scalebar_label=True)

        # unknown channel label
        with pytest.raises(Exception) as e_info:
            img2d.plot(channels_labels=['error'],
                       pyramid_level_coord=2)

        # both channels and channels_labels given
        with pytest.warns(UserWarning):
            img2d.plot(channels=[0],
                       channels_labels=['some-label-2'],
                       scalebar_micrometer=50,
                       show_scalebar_label=False)

        # automatically extract channel colors and ranges
        img2d.plot(channels=[0])

        # using label_name
        img2d.plot(label_name='organoids',
                   channels=[0],
                   channel_colors=['white'],
                   channel_ranges=[[0.01, 0.99]])

        # automatically extract coordinates for label_value
        with pytest.warns(UserWarning):
            img2d.plot(upper_left_yx=(0, 0),
                       lower_right_yx=(100, 100),label_name='organoids',
                       label_value=3,
                       padding_pixels=8)
        
        # ... restrict plotting to label_value
        img2d.plot(label_name='organoids',
                   label_value=3,
                   restrict_to_label_values=[3])
        
        # ... restrict plotting to non-existent label_value
        img2d.plot(label_name='organoids',
                   label_value=3,
                   restrict_to_label_values=99)
        
        # ... non-existent label_value
        with pytest.raises(Exception) as e_info:
            img2d.plot(label_name='organoids',
                       label_value=99)

        # using label_name without matching pyrmaid level
        # ... copy zarr fileset
        assert tmpdir.check()
        shutil.copytree('tests/example_data/plate_ones_mip.zarr/B/03/0',
                        str(tmpdir) + '/example_img')
        assert tmpdir.join('/example_img/1').check()
        # ... remove pyramid levels 1 and 2 from image
        shutil.rmtree(str(tmpdir) + '/example_img/1')
        shutil.rmtree(str(tmpdir) + '/example_img/2')
        zattr_file = str(tmpdir) + '/example_img/.zattrs'
        with open(zattr_file) as f:
            zattr = json.load(f)
        zattr['multiscales'][0]['datasets'] = [zattr['multiscales'][0]['datasets'][0]]
        with open(zattr_file, "w") as jsonfile:
            json.dump(zattr, jsonfile, indent=4)
        # ... remove pyramid levels 0 and 1 from label organoids
        shutil.rmtree(str(tmpdir) + '/example_img/labels/organoids/0')
        shutil.rmtree(str(tmpdir) + '/example_img/labels/organoids/1')
        zattr_file = str(tmpdir) + '/example_img/labels/organoids/.zattrs'
        with open(zattr_file) as f:
            zattr = json.load(f)
        zattr['multiscales'][0]['datasets'] = [zattr['multiscales'][0]['datasets'][2]]
        with open(zattr_file, "w") as jsonfile:
            json.dump(zattr, jsonfile, indent=4)
        # ... plot
        imgtmp = ome_zarr.Image(str(tmpdir) + '/example_img')
        imgtmp.plot(label_name='organoids')
        # ... clean up
        shutil.rmtree(str(tmpdir) + '/example_img')
        ### TODO: problematic case where pyramid_level_coord is not None
        ###       and pyramid_level is not parallel in image and label
        ###       what does pyramid_level_coord refer to?
        ###       solution: remove pyramid_level_coord argument?
        ###       not needed: can use coordinate_unit="micrometer"

    plt.savefig(tmpdir.join('output.png'))
    assert True # check if the plotting ran through
