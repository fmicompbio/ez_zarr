# for testing, run the following from the project folder:
#     pip install -e .
#     pytest --color=yes -v --cov=./src --cov-report=term-missing tests

import pytest
import numpy as np

from ez_zarr import utils

# coordinate conversion ---------------------------------------------
def test_convert_coordinates():
    """Test `ome_zarr.utils.convert_coordinates() function."""
    assert utils.convert_coordinates(
        (10, 30), [1, 1], [2, 3]) == (5, 10)
    assert utils.convert_coordinates(
        (5, 10), [2, 3], [1, 1]) == (10, 30)
    assert utils.convert_coordinates(
        (5, 10, 20), [1, 0.2, 0.2], [1, 1, 1]) == (5, 2, 4)

# image rescaling and resizing --------------------------------------
def test_rescale_image():
    """Test `ome_zarr.utils.rescale_image() function."""
    im1 = np.zeros((10, 10))
    im2 = np.random.rand(3, 20, 20)
    im3 = np.random.randint(0, 10, (3, 20, 20))

    # invalid input
    with pytest.raises(Exception) as e_info:
        utils.rescale_image(im1, [1], [2, 2])
    with pytest.raises(Exception) as e_info:
        utils.rescale_image(im1, [1, 1], None)
    with pytest.raises(Exception) as e_info:
        utils.rescale_image(im1, [1, 1], [2, 2], 'error')
    with pytest.raises(Exception) as e_info:
        utils.rescale_image(im1, [1, 1], [2, 2], 'intensity', 4)
    
    # expected results
    assert np.all(utils.rescale_image(im1, [2, 2], [1, 1]) == np.zeros((20, 20)))
    assert np.all(utils.rescale_image(im1, [2, 2], [4, 9]) == np.zeros((5, 2)))
    assert np.all(utils.rescale_image(im2, [1, 3.5, 7.7], [1, 3.5, 7.7], number_nonspatial_axes=1) == im2)
    with pytest.warns(UserWarning):
        im3s = utils.rescale_image(im3, [1, 1, 1], [2, 2, 2], number_nonspatial_axes=1, im_type='label')
    assert np.all(np.unique(im3s) in np.unique(im3))

def test_resize_image():
    """Test `ome_zarr.utils.resize_image() function."""
    im1 = np.zeros((10, 10))
    im2 = np.random.rand(3, 20, 20)
    im3 = np.random.randint(0, 10, (3, 20, 20))

    # invalid input
    with pytest.raises(Exception) as e_info:
        utils.resize_image(im1, (20, 20, 20))
    with pytest.raises(Exception) as e_info:
        utils.resize_image(im1, (20, 20), 'error')
    with pytest.raises(Exception) as e_info:
        utils.resize_image(im1, (20, 20), 'intensity', 4)
    
    # expected results
    assert np.all(utils.resize_image(im1, (20, 20)) == np.zeros((20, 20)))
    assert np.all(utils.resize_image(im1, (5, 2)) == np.zeros((5, 2)))
    assert np.all(utils.resize_image(im2, im2.shape, number_nonspatial_axes=1) == im2)
    with pytest.warns(UserWarning):
        im3s = utils.resize_image(im3, (6, 30, 30), number_nonspatial_axes=1, im_type='label')
    assert np.all(np.unique(im3s) in np.unique(im3))
