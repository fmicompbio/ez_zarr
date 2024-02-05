"""Helpful functions for plotting and working with image arrays."""

__all__ = ['zproject']
__version__ = '0.1.3'
__author__ = 'Silvia Barbiero, Michael Stadler'


# imports ---------------------------------------------------------------------
import dask.array
import numpy as np
from typing import Union, Optional
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# global variables ------------------------------------------------------------
plate_layouts = {
    '6well': {'rows': ['A','B'],
              'columns': [str(i+1).zfill(2) for i in range(3)]},
    '24well': {'rows': ['A','B','C','D'],
               'columns': [str(i+1).zfill(2) for i in range(6)]},
    '96well': {'rows': ['A','B','C','D','E','F','G','H'],
               'columns': [str(i+1).zfill(2) for i in range(12)]},
    '384well': {'rows': ['A','B','C','D','E','F','G','H',
                         'I','J','K','L','M','N','O','P'],
                'columns': [str(i+1).zfill(2) for i in range(24)]},
}

def get_shuffled_cmap(cmap_name: str='hsv', seed: int=42) -> mcolors.ListedColormap:
    """
    Create shuffled color map.

    Parameters:
        cmap_name (str): The name of the color map (passed to `cmap` argument
            of matplotlib.colormaps.get_cmap).
        seed (int): Used to seed the random number generator with
            numpy.random.seed.
    
    Examples:
        Obtain a matplotlib.colors.ListedColormap with randomized color ordering:

        >>> cm = plotting.get_shuffled_cmap()
    """
    cmap = plt.colormaps.get_cmap(cmap_name)
    all_colors = cmap(np.linspace(0, 1, cmap.N))
    np.random.seed(seed)
    np.random.shuffle(all_colors)
    shuffled_cmap = mcolors.ListedColormap(all_colors)
    return shuffled_cmap

# helper functions to manipulate image arrays ---------------------------------
def zproject(im: Union[dask.array.Array, np.ndarray],
             axis: Optional[int]=1,
             method: Optional[str]='maximum',
             keepdims: Optional[bool]=True,
             img_bit: Optional[int]=16) -> Union[dask.array.Array, np.ndarray]:
    """
    Project a 4D or 3D image along z-axis according to desired projection method.

    Helper function to conveniently project multi-dimensional images along z-axis according to desired projection method. 

    Parameters:
        im (dask.array or np.ndarray): The array to be projected
        axis (int): The axis on which to perform the projection (typically
            the z-axis). Unless otherwise specified, it defaults to axis 1 (by convention Fractal outputs have shape ch,z,y,x). Note: if masks are projected, Fractal output have usually shape z,y,x, so use `axis=0` in that case.
        method (str): The projection method of choice. Available are 
            'maximum' (default), 'minimum' (typically used for brightfield images), 'sum', 'average'
        keepdims (bool): Should the output array keep the same dimensions 
            as the input array, or should the z-axis been squeezed (default behaviour of e.g. np.min())
        img_bit (int): Is the image 8- or 16-bit? Relevant for 'sum' 
            projection, where clipping might be necessary
    
    Returns:
        The input array where the z-axis has been projected according to the method of choice.
    
    Examples:
        Project np.array along z-axis:

        >>> zproject(im=np.array([[[1,2,4],[1,3,5],[1,6,7]],[[1,2,4],[1,3,5],[1,6,7]]]), method='maximum', axis=1)
    """
    methods = {
        'maximum': np.max,
        'minimum': np.min,
        'sum': np.sum,
        'average': np.mean,
    }

    if not method in methods:
        raise ValueError(f"Unknown method ({method}), should be one of: " + ', '.join(list(methods.keys())))
    
    im = methods[method](im, axis=axis, keepdims=keepdims) 
    if method == 'sum':
        max_value = (2**img_bit) - 1
        im = np.clip(im, 0, max_value)

    return im

def pad_image(im: Union[dask.array.Array, np.ndarray],
              output_shape: tuple,
              constant_value: int=0) -> np.ndarray:
    """
    Pad an image by adding pixels of `constant_value` symmetrically around it
    to make it shape `output_shape`.

    Parameters:
        im (dask.array or numpy.ndarray): Input image.
        output_shape (tuple[int]): Desired output shape after padding (must be greater
            or equal to im.shape).
        constant_value (int): Value for added pixels.
    
    Returns:
        np.ndarray with padded image of shape `output_shape`.
    
    Examples:
        Make the image `im` (100, 100) in shape:

        >>> pad_image(im, (100, 100))
    """
    # digest arguments
    assert len(im.shape) == len(output_shape), f"output_shape {output_shape} should be of length {len(im.shape)}"
    assert all([im.shape[i] <= output_shape[i] for i in range(len(output_shape))]), f"output_shape {output_shape} must be greater or equal to image shape {im.shape}"
    assert isinstance(constant_value, int)

    # calculate padding size
    pad_total = np.subtract(output_shape, im.shape)
    pad_before = [int(x) for x in pad_total / 2]
    pad_after = pad_total - pad_before
    pad_size = tuple([(pad_before[i], pad_after[i]) for i in range(len(pad_total))])
    
    # add padding
    im = np.pad(im, pad_width=pad_size, mode='constant', constant_values=0)
    return im

def convert_to_rgb(im: Union[dask.array.Array, np.ndarray],
                   channel_colors: list[str]=['white'],
                   channel_ranges: list[list[float]]=[[0.01, 0.99]]) -> np.ndarray:
    """
    Convert a (channels, y, x) array to an RGB array.
    
    Parameters:
        im (dask.array or numpy.ndarray): Input image of shape (channels, y, x).
        channel_colors (list[str]): A list with python color strings (e.g. 'red')
            corresponding with the color for each channel in `im`.
        channel_ranges (list[list[float]]): A list of 2-element lists (e.g. [0.01, 0.99])
            giving the value ranges that should be mapped to colors for each
            channel. If the given numerical values are less or equal to 1.0,
            they are interpreted as quantiles and the corresponding intensity
            values are calculated on the channel non-zero values, otherwise they
            are directly used as intensities. Values outside of this range will
            be clipped.

    Returns:
        An RGB image of shape (x, y, 3), where the third axis corresponds to
            red, green and blue channels. The image is suitable for plotting with
            matplotlib.pyplot.imshow.

    Examples:
        Convert an image `img` from (2, y, x) to (x, y, 3), using the
        channels as yellow and blue intensities:

        >>> img_rgb = convert_to_rgb(img, ['yellow', 'blue'])   
    """
    # digest arguments
    assert len(im.shape) == 3, f"`im` has shape {im.shape} but must have three dimensions"
    assert len(channel_colors) == im.shape[0], (
        f"`channel_colors` must be of the same length as the first axis in `im` ({im.shape[0]})"
    )
    assert len(channel_ranges) == len(channel_colors), (
        f"`channel_ranges` must be of the same length as `channel_colors` ({len(channel_colors)})"
    )

    # clip and normalize each channel according to the channel_ranges (im: (ch,y,x))
    ranges_used = []
    for i in range(len(channel_ranges)):
        if max(channel_ranges[i]) <= 1.0:
            # ranges are given as quantiles -> calculate corresponding intensities
            nonzero = im[i] > 0
            rng = np.quantile(a=im[i][nonzero], q=channel_ranges[i], overwrite_input=False)
        else:
            rng = channel_ranges[i]
        ranges_used.append(rng)
        im[i] = np.clip(im[i], rng[0], rng[1])
        im[i] = (im[i] - rng[0]) / (rng[1] - rng[0])

    # convert the color strings to RGB values
    rgb_colors = np.array([mcolors.to_rgb(c) for c in channel_colors])    

    # multiply channels with channel_colors and create weighted sum to (x,y,3)
    rgb_im = np.einsum('cyx,cr->xyr', im, rgb_colors)

    # clip to [0,1], map to [0, 255] and convert to uint8 type
    rgb_im = np.clip(rgb_im, 0, 1)
    rgb_im = (rgb_im * 255.0).astype(np.uint8)

    return rgb_im
