import pytest
import os
# for testing without installing easy_ome_zarr, use the following call
# from the folder containing `easy_ome_zarr`:
#     python -m pytest easy_ome_zarr/tests/

from easy_ome_zarr import zarr_wraps

@pytest.fixture
def example_plate_3d():
    return zarr_wraps.FmiZarr('easy_ome_zarr/tests/example_data/plate_ones.zarr')

@pytest.fixture
def example_plate_2d():
    return zarr_wraps.FmiZarr('easy_ome_zarr/tests/example_data/plate_ones_mip.zarr')

def test_constructor_3d(example_plate_3d):
    assert isinstance(example_plate_3d, zarr_wraps.FmiZarr)
    assert example_plate_3d.name == 'plate_ones.zarr'

def test_constructor_2d(example_plate_2d):
    assert isinstance(example_plate_2d, zarr_wraps.FmiZarr)
    assert example_plate_2d.name == 'plate_ones_mip.zarr'
