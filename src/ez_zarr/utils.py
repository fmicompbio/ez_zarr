"""Utility functions for `ez_zarr`."""

__all__ = ['convert_coordinates']
__version__ = '0.2.0'
__author__ = 'Silvia Barbiero, Michael Stadler, Charlotte Soneson'


# imports -------------------------------------------------------------------
import numpy as np
from typing import Union, Optional, Any

# coordinate conversion ---------------------------------------------------
def convert_coordinates(coords_from: tuple[Union[int, float]],
                        scale_from: list[float],
                        scale_to: list[float]) -> tuple[Union[int, float]]:
    """
    Convert coordinates between scales.

    Parameters:
        coords_from (tuple): The coordinates to be converted.
        scale_from (list[float]): The scale that the coordinates refer to.
            Needs to be parallel to `coords_from`.
        scale_to (list[float]): The scale that the coordinates should be
            converted to.
    
    Returns:
        A tuple of the same length as `coords_from` with coordinates in
        the new scale.
    
    Examples:
        Convert a point (10, 30) from [1, 1] to [2, 3]:

        >>> from ez_zarr.utils import convert_coordinates
        >>> y, x = convert_coordinates((10,30), [1, 1], [2, 3])
    """
    # digest arguments
    assert len(coords_from) == len(scale_from)
    assert len(coords_from) == len(scale_to)

    # convert and return
    coords_to = np.divide(np.multiply(coords_from, scale_from), scale_to)
    return(tuple(coords_to))

