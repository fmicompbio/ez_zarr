"""Utility functions for `ez_zarr`."""

__all__ = ['convert_coordinates', 'rescale_image', 'resize_image']
__version__ = '0.2.0'
__author__ = 'Silvia Barbiero, Michael Stadler, Charlotte Soneson'


# imports -----------------------------------------------------------
import numpy as np
from skimage.transform import rescale, resize
import warnings
from typing import Union, Optional, Any

# coordinate conversion ---------------------------------------------
def convert_coordinates(coords_from: tuple[Union[int, float], ...],
                        scale_from: list[float],
                        scale_to: list[float]) -> tuple[Union[int, float], ...]:
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

# image rescaling and resizing --------------------------------------
def _setup_for_resizing(im: np.ndarray,
                        im_type: str,
                        number_nonspatial_axes: int,
                        scale_from: Optional[list[float]]=None,
                        scale_to: Optional[list[float]]=None,
                        output_shape: Optional[tuple[int]]=None) -> tuple[list[int], np.ndarray, int, bool]:
    # Digest arguments
    assert isinstance(im, np.ndarray), "`im` must be a numpy array"
    assert scale_from is None or len(scale_from) == im.ndim, f"`scale_from` must be of length {im.ndim}"
    assert scale_to is None or len(scale_to) == im.ndim, f"`scale_to` must be of length {im.ndim}"
    assert output_shape is None or len(output_shape) == im.ndim, f"`output_shape` must be of length {im.ndim}"
    assert im_type in ['intensity', 'label'], f"invalid `im_type` '{im_type}' - must be 'intensity' or 'label'"
    assert isinstance(number_nonspatial_axes, int) and number_nonspatial_axes >= 0 and number_nonspatial_axes < im.ndim, f"`number_of_nonspatial_axes` must be an integer greater or equal to 0 and less than the number of image dimensions ({im.ndim})"

    # Get the spatial and non-spatial axes
    non_spatial_axes = list(range(number_nonspatial_axes))
    spatial_axes = list(range(number_nonspatial_axes, im.ndim))
    if scale_from is not None and scale_to is not None and any([scale_from[i] != scale_to[i] for i in non_spatial_axes]):
        warnings.warn("Ignoring `scale_from` and `scale_to` components for non-spatial axes.")
    if output_shape is not None and any([im.shape[i] != output_shape[i] for i in non_spatial_axes]):
        warnings.warn("Ignoring `output_shape` components for non-spatial axes.")

    # Reshape the image to separate non-spatial from spatial axes
    reshaped_im = im.reshape(int(np.prod([im.shape[i] for i in non_spatial_axes])),
                            *[im.shape[i] for i in spatial_axes])
    
    # Set scaling/resizing parameters
    if im_type == 'intensity':
        interpolation_order = 1
        do_anti_aliasing = True
    elif im_type == 'label':
        interpolation_order = 0
        do_anti_aliasing = False
    
    # return results
    return (non_spatial_axes, reshaped_im, interpolation_order, do_anti_aliasing)

def rescale_image(im: np.ndarray,
                  scale_from: list[float],
                  scale_to: list[float],
                  im_type: str='intensity',
                  number_nonspatial_axes: int=0) -> np.ndarray:
    """
    Rescale an image (2 to 5-dimensional arrays, possibly with non-spatial axes).

    Parameters:
        im (numpy.ndarray): The image to be rescaled.
        scale_from (list[float]): The scale that the image refers to.
            Needs to be parallel to `im`.
        scale_to (list[float]): The scale that the image should be
            converted to. Needs to be of the same length as `scale_from`.
            The scaling factors are calculated as the ratio between
            `scale_from` and `scale_to`.
        im_type (str): The type of the image. Can be
            'intensity' (default) or 'label'. An intensity image is
            rescaled with anti-aliasing, while a label image is rescaled
            without, so that the resulting image only contains values
            (labels) also contained in the input.
        number_nonspatial_axes (int): Number of first axes that refer to
            non-spatial dimensions, such as time or channel. Rescaling is
            performed only on the spatial axes, separately for each value
            or combination of values of the non-spatial axes.
    
    Returns:
        The rescaled image as a `numpy.ndarray`.
    
    Examples:
        Rescale a 4D image (channels, z, y, x):

        >>> im_large = np.random.rand(4, 100, 100)
        >>> im_small = rescale_image(im_large, [1, 0.3, 0.3], [1, 0.6, 0.6], number_nonspatial_axes=1)
        >>> im_small.shape # returns (4, 50, 50)
    """

    # setup for rescale
    non_spatial_axes, reshaped_im, interpolation_order, do_anti_aliasing = _setup_for_resizing(
        im=im,
        scale_from=scale_from,
        scale_to=scale_to,
        im_type=im_type,
        number_nonspatial_axes=number_nonspatial_axes)

    # Calculate the rescaling factors
    scale_factors = [from_ / to for from_, to in zip(scale_from, scale_to)]

    # Rescale the spatial dimensions separately for each non-spatial element
    rescaled_im = np.stack([rescale(image=reshaped_im[i],
                                    scale=scale_factors[number_nonspatial_axes:],
                                    order=interpolation_order,
                                    preserve_range=True,
                                    anti_aliasing=do_anti_aliasing)
                            for i in range(reshaped_im.shape[0])])
    # REMARK: is it necessary to rescale the label differently, like this:
    # ms_small = np.zeros_like(im_small[0,:,:,:], dtype=np.int32)
    # for l in filter(None, np.unique(ms)):
    #     ms_small[rescale(image=ms==l, scale=scale_factor, order=0, anti_aliasing=False)] = l


    # Reshape the rescaled array to its original shape
    rescaled_im = rescaled_im.reshape(*[[im.shape[i] for i in non_spatial_axes] +
                                        list(rescaled_im.shape[1:])])
    
    return rescaled_im

def resize_image(im: np.ndarray,
                 output_shape: tuple[int],
                 im_type: str='intensity',
                 number_nonspatial_axes: int=0) -> np.ndarray:
    """
    Resize an image (2 to 5-dimensional arrays, possibly with non-spatial axes).

    Parameters:
        im (numpy.ndarray): The image to be rescaled.
        output_shape (tuple[int]): The shape of the output image. This must
            be parallel to the shape of the input image.
        im_type (str): The type of the image. Can be
            'intensity' (default) or 'label'. An intensity image is
            rescaled with anti-aliasing, while a label image is rescaled
            without, so that the resulting image only contains values
            (labels) also contained in the input.
        number_nonspatial_axes (int): Number of first axes that refer to
            non-spatial dimensions, such as time or channel. Rescaling is
            performed only on the spatial axes, separately for each value
            or combination of values of the non-spatial axes.
    
    Returns:
        The rescaled image as a `numpy.ndarray`.
    
    Examples:
        Resize a 4D image (channels, z, y, x):

        >>> im_large = np.random.rand(4, 100, 100)
        >>> im_small = resize_image(im_large, (4, 50, 50), number_nonspatial_axes=1)
        >>> im_small.shape # returns (4, 50, 50)
    """

    # setup for rescale
    non_spatial_axes, reshaped_im, interpolation_order, do_anti_aliasing = _setup_for_resizing(
        im=im,
        output_shape=output_shape,
        im_type=im_type,
        number_nonspatial_axes=number_nonspatial_axes)

    # Rescale the spatial dimensions separately for each non-spatial element
    resized_im = np.stack([resize(image=reshaped_im[i],
                                  output_shape=output_shape[number_nonspatial_axes:],
                                  order=interpolation_order,
                                  preserve_range=True,
                                  anti_aliasing=do_anti_aliasing)
                            for i in range(reshaped_im.shape[0])])
    # REMARK: is it necessary to rescale the label differently, like this:
    # ms_small = np.zeros_like(im_small[0,:,:,:], dtype=np.int32)
    # for l in filter(None, np.unique(ms)):
    #     ms_small[rescale(image=ms==l, scale=scale_factor, order=0, anti_aliasing=False)] = l


    # Reshape the resized array to its original shape
    resized_im = resized_im.reshape(*[[im.shape[i] for i in non_spatial_axes] +
                                      list(resized_im.shape[1:])])
    
    return resized_im
