# for testing, run the following from the project folder:
#     pip install -e .
#     pytest --color=yes -v --cov=./src --cov-report=term-missing tests

import pytest

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
