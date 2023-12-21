# for testing without installing easy_ome_zarr, use the following call
# from the folder containing `easy_ome_zarr`:
#     python -m pytest easy_ome_zarr/tests/
#     python -m pytest --cov easy_ome_zarr/tests/

import pytest
import zarr
import pandas as pd
import anndata as ad
from easy_ome_zarr import zarr_wraps

@pytest.fixture
def plate_3d():
    return zarr_wraps.FmiZarr('easy_ome_zarr/tests/example_data/plate_ones.zarr')

@pytest.fixture
def plate_2d():
    return zarr_wraps.FmiZarr('easy_ome_zarr/tests/example_data/plate_ones_mip.zarr')

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
    assert plate_2d.name == 'plate_ones_mip.zarr'
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
    assert plate_2d.label_names == []
    assert plate_2d.table_names == ['FOV_ROI_table']

def test_get_table_2d(plate_2d):
    empty = plate_2d.get_table('does not exist')
    df = plate_2d.get_table('FOV_ROI_table')
    ann = plate_2d.get_table('FOV_ROI_table', as_AnnData = True)
    assert empty is None
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (4, 8)
    assert isinstance(ann, ad.AnnData)
    assert ann.shape == (4, 8)
