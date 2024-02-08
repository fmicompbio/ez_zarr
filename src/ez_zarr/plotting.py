"""Helpful functions for plotting and working with image arrays."""

__all__ = ['zproject']
__version__ = '0.1.4'
__author__ = 'Silvia Barbiero, Michael Stadler'


# imports ---------------------------------------------------------------------
from typing import Union, Optional
from copy import deepcopy
import dask.array
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.ticker as ticker

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
    Convert a (channels, y, x) array to an RGB array of shape (y, x, 3).
    
    This function will return a copy of `im` in order not to affect the input.

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
        An RGB image of shape (y, x, 3), where the third axis corresponds to
            red, green and blue channels. The image is suitable for plotting with
            `matplotlib.pyplot.imshow` and oriented such that is results in the
            same orentiation as when directly plotting a single channel.

    Examples:
        Convert an image `img` from (2, y, x) to (y, x, 3), using the
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

    # make a copy of the input argument
    if isinstance(im, dask.array.Array):
        im = im.compute()
    im = deepcopy(im).astype(np.float64)

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

    # multiply channels with channel_colors and create weighted sum to (y,x,3)
    rgb_im = np.einsum('cyx,cr->yxr', im, rgb_colors)

    # clip to [0,1], map to [0, 255] and convert to uint8 type
    rgb_im = np.clip(rgb_im, 0, 1)
    rgb_im = (rgb_im * 255.0).astype(np.uint8)

    return rgb_im

# helper functions to plot image arrays ---------------------------------------
def plot_image(im: np.ndarray,
               msk: Optional[np.ndarray]=None,
               msk_alpha: float=0.3,
               channels: list[int]=[0],
               channel_colors: list[str]=['white'],
               channel_ranges: list[list[float]]=[[0.01, 0.95]],
               z_projection_method: str='maximum',
               pad_to_yx: list[int]=[0, 0],
               pad_value: int=0,
               axis_style: str='none',
               spacing_yx: Optional[list[float]]=None,
               title: Optional[str]=None,
               scalebar_pixel: int=0,
               scalebar_color: str='white',
               scalebar_position: str='bottomright',
               call_show: bool=True,
               fig_width_inch: float=24.0,
               fig_height_inch: float=16.0,
               fig_dpi: int=150,
               fig_style: str='dark_background'):
        """
        Plot an image array, optionally overlaid with a segmentation mask.
         
        Plot an image (provided as an array with shape (ch,z,y,x) or (ch,y,x))
        by summarizing multiple z planes using `z_projection_method` (if needed),
        mapping channels values (in the range of `channel_ranges`) to colors
        (given by `channel_colors`), converting it to an RGB array of shape
        (x,y,3) and plotting it using `matplotlib.pyplot.imshow`.
        If `msk` is provided, it is assumed to contain a compatible segmentation
        mask and will be plotted transparently (controlled by `msk_alpha`)
        on top of the image.

        Parameters:
            im (numpy.ndarray): An array with four (ch,z,y,x) or three (ch,y,x)
                dimensions with image intensities.
            msk (numpy.ndarray): An optional cooresponding mask array with
                shape `im.shape[1:]`. If given, label values will be mapped
                to discrete colors and plotted transparently on top of the image.
            msk_alpha (float): A scalar value between 0 (fully transparent)
                and 1 (solid) defining the transparency of the mask.
            channels (list[int]): The image channel(s) to be plotted. For example,
                to plot the first and third channel of a 4-channel image with
                shape (4,1,500,600), you can use `channels=[0, 2]`.
            channel_colors (list[str]): A list with python color strings
                (e.g. 'red') defining the color for each channel in `channels`.
                For example, to map the selected `channels=[0, 2]` to
                cyan and magenta, respectively, you can use
                `channel_colors=['cyan', 'magenta']`.
            channel_ranges (list[list[float]]): A list of 2-element lists
                (e.g. [0.01, 0.95]) giving the the value ranges that should
                be mapped to colors for each channel. If the given numerical
                values are less or equal to 1.0, they are interpreted as quantiles
                and the corresponding intensity values are calculated on the
                channel non-zero values, otherwise they are directly used as
                intensities. Values outside of this range will be clipped.
            z_projection_method (str): Method for combining multiple z planes.
                For available methods, see ez_zarr.plotting.zproject
                (default: 'maximum').
            pad_to_yx (list[int]): If the image or label mask are smaller, pad
                them by `pad_value` to this total y and x size. 
            pad_value (int): value to use for contant-value image padding.
            axis_style (str): A string scalar defining how to draw the axis. Should
                be one of 'none' (no axis, the default), 'pixel' (show pixel labels),
                'frame' (show just a frame around the plot without ticks)
                or 'micrometer' (show micrometer labels). If `axis_style='micrometer'`,
                `spacing_yx` is used to convert pixel to micrometer.
            spacing_yx (list[float]): The spacing of pixels in y and x direction,
                in micrometer, used to convert pixels to micrometer when
                `axis_style='micrometer'`.
            title (str): String to add as a title on top of the image. If `None`
                (the default), no title will be added.
            scalebar_pixel (int): If non-zero, draw a scalebar at the lower right
                corner of the image of size `scalebar_pixel`.
            scalebar_color (str): Scalebar color.
            scalebar_position (str): position of the scale bar, one of 'topleft',
                'topright', 'bottomleft' or 'bottomright'
            call_show (bool): If true, the call to `matplotlib.pyplot.imshow` is
                embedded between `matplotlib.pyplot.figure` and
                `matplotlib.pyplot.show`/`matplotlib.pyplot.close` calls.
                This is the default behaviour and typically used when an indiviual
                image should be plotted and displayed. It can be set to `False`
                if multiple images should be plotted and their arrangement
                is controlled outside of `plotting.plot_image`. The parameters
                `fig_width_inch`, `fig_height_inch`, `fig_dpi` and `fig_style`
                are ignored in that case.
            fig_width_inch (float): Figure width (inch).
            fig_height_inch (float): Figure height (inch).
            fig_dpi (int): Figure resolution (dots per inch).
            fig_style (str): Style for the figure. Supported are 'brightfield', which
                is a special mode for single-channel brightfield microscopic images
                (it will automatically set `channels=[0]`, `channel_colors=['white']`
                `z_projection_method='minimum'`, `pad_value=1` and `fig_style='default'`),
                and any other styles that can be passed to `matplotlib.pyplot.style.context`
                (default: 'dark_background')

        Examples:
            Plot the first channel of `img` in gray-scale.

            >>> plot_image(img, channels=[0], channel_colors=['white'])
        """
        # digest arguments
        ndim = len(im.shape)
        assert ndim == 3 or ndim == 4, (
            f"Unsupported image dimension ({im.shape}), ",
            "should be either (ch,z,y,x) or (ch,y,x)"
        )
        assert msk is None or msk.shape == im.shape[1:], (
            f"`msk` dimension ({msk.shape}) is not compatible with `im` "
            f"dimension, should be ({im.shape[1:]})"
        )
        nch = im.shape[0]
        assert all([ch <= nch for ch in channels]), (
            f"Invalid `channels` parameter, must be less or equal to {nch}"
        )
        assert axis_style != 'micrometer' or spacing_yx != None, (
            f"For `axis_style='micrometer', the parameter `spacing_yx` needs to be provided."
        )

        # adjust parameters for brightfield images
        if fig_style == 'brightfield':
            channels = [0]
            channel_colors = ['white']
            pad_value = 1
            z_projection_method = 'minimum'
            fig_style = 'default'

        # combine z planes if needed
        if ndim == 4:
            # im: (ch,z,y,x) -> (ch,y,x))
            im = zproject(im=im, method=z_projection_method,
                          axis=1, keepdims=False)
            if not msk is None:
                # msk: (z,y,x) -> (y,x)
                msk = zproject(im=msk, method='maximum', # always use 'maximum' for labels
                            axis=0, keepdims=False)

        # pad if necessary
        if any([v > 0 for v in pad_to_yx]) and (im.shape[1] <= pad_to_yx[0] and im.shape[2] <= pad_to_yx[1]):
            im = pad_image(im=im,
                           output_shape=[nch, pad_to_yx[0], pad_to_yx[1]],
                           constant_value=pad_value)
            if not msk is None:
                msk = pad_image(im=msk,
                                output_shape=pad_to_yx,
                                constant_value=pad_value)
        
        # convert (ch,y,x) to rgb (y,x,3) and plot
        im_rgb = convert_to_rgb(im=im[channels],
                                channel_colors=channel_colors,
                                channel_ranges=channel_ranges)

        # define nested function with main plotting code
        def _do_plot():
            plt.imshow(im_rgb)
            if not msk is None and np.max(msk) > 0:
                plt.imshow(msk,
                           interpolation='none',
                           cmap=get_shuffled_cmap(),
                           alpha=np.multiply(msk_alpha, msk > 0))
            if axis_style == 'none':
                plt.axis('off')
            elif axis_style == 'pixel':
                pass
            elif axis_style == 'frame':
                plt.xticks([]) # remove axis ticks
                plt.yticks([])
            elif axis_style == 'micrometer':
                ax = plt.gca() # get current axes
                yticks = ticker.FuncFormatter(lambda y, pos: '{0:g}'.format(y * spacing_yx[0]))
                xticks = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x * spacing_yx[1]))
                ax.yaxis.set_major_formatter(yticks)
                ax.xaxis.set_major_formatter(xticks)
            else:
                raise ValueError(f"Unknown `axis_style` ({axis_style}), should be one of 'none', 'pixel', 'frame' or 'micrometer'")
            if title != None:
                plt.title(title)
            if scalebar_pixel != 0:
                img_yx = im_rgb.shape[0:2]
                d = min([round(img_yx[i] * 0.05) for i in range(2)]) # 5% margin
                scalebar_height = round(img_yx[0] * 0.01)
                if scalebar_position == 'bottomright':
                    pos_xy = (img_yx[1] - d - scalebar_pixel,
                              img_yx[0] - d)
                elif scalebar_position == 'bottomleft':
                    pos_xy = (d,
                              img_yx[0] - d)
                elif scalebar_position == 'topleft':
                    pos_xy = (d,
                              d - scalebar_height)
                elif scalebar_position == 'topright':
                    pos_xy = (img_yx[1] - d - scalebar_pixel,
                              d - scalebar_height)
                else:
                    raise ValueError(f"Unknown scalebar_position ({scalebar_position}), should be one of 'bottomright', 'bottomleft', 'topright', or 'topleft'")
                rect = patches.Rectangle(xy=pos_xy, width=scalebar_pixel,
                                         height=scalebar_height,
                                         edgecolor=scalebar_color,
                                         facecolor=scalebar_color)
                ax = plt.gca() # get current axes
                ax.add_patch(rect) # add the patch to the axes

        # create the plot
        if call_show:
            with plt.style.context(fig_style):
                fig = plt.figure(figsize=(fig_width_inch, fig_height_inch))
                fig.set_dpi(fig_dpi)
                _do_plot()
                plt.show()
                plt.close()
        else:
            _do_plot()
