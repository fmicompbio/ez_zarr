"""Wrap OME-Zarr filesets at different levels in classes.

Represent an OME-Zarr fileset as a class to give high-level
access to its contents.

Classes:
    Image: Contains a single `.zgroup`, typicallly a single image and possibly derived labels or tables.
"""

__all__ = ['Image']
__version__ = '0.2.0'
__author__ = 'Silvia Barbiero, Michael Stadler, Charlotte Soneson'


# imports -------------------------------------------------------------------
import os
import numpy as np
import zarr
import dask.array
import pandas as pd
import importlib
import warnings
import random
from typing import Union, Optional, Any


# Image class -----------------------------------------------------------------
class Image:
    """Represents an OME-Zarr image."""

    # constructor and helper functions ----------------------------------------
    def __init__(self, path: str,
                 name: Optional[str]=None,
                 skip_checks: Optional[bool]=False) -> None:
        """
        Initializes an OME-Zarr image from its path, containing a
        single zarr group, possibly with multiple resolution levels,
        derived labels or tables, but no further groups.

        Parameters:
            path (str): Path containing the OME-Zarr image.
            name (str, optional): Optional name for the image.
            skip_checks (bool, optional): If `True`, no checks are performed
        
        Examples:
            Get an object corresponding to an image.

            >>> from ez_zarr import ome_zarr
            >>> imageA = ome_zarr.Image('path/to/image')
        """

        self.path: str = path
        self.name: str = ''
        if name:
            self.name = name
        else:
            self.name = os.path.basename(self.path)
        self.zarr_group: zarr.Group = zarr.open_group(store=self.path, mode='r')
        self.array_dict = {x[0]: x[1] for x in self.zarr_group.arrays()}
        self.ndim = self.array_dict[list(self.array_dict.keys())[0]].ndim
        self.label_names = []
        if 'labels' in list(self.zarr_group.group_keys()):
            self.label_names = [x for x in self.zarr_group.labels.group_keys()]
        self.table_names = []
        if 'tables' in list(self.zarr_group.group_keys()):
            self.table_names = [x for x in self.zarr_group.tables.group_keys()]

        if not skip_checks:
            # make sure that it does not contain any further groups
            if len([x for x in self.zarr_group.group_keys() if x not in ['labels', 'tables']]) > 0:
                raise ValueError(f"{self.path} contains further groups")
            
        # load info about available scales in image and labels
        if not 'multiscales' in self.zarr_group.attrs:
            raise ValueError(f"{self.path} does not contain a 'multiscales' attribute")
        self.multiscales_image = self._load_multiscale_info(self.zarr_group, skip_checks)
        self.multiscales_labels = {x: self._load_multiscale_info(self.zarr_group.labels[x], skip_checks) for x in self.label_names}
        self.axes_unit_image = self._load_axes_unit(self.multiscales_image)
        self.axes_unit_labels = {x: self._load_axes_unit(self.multiscales_labels[x]) for x in self.label_names}

        # load channel metadata
        # ... label dimensions, e.g. "czyx"
        self.channel_info_image = self._load_channel_info(self.multiscales_image)
        self.channel_info_labels = {x: self._load_channel_info(self.multiscales_labels[x]) for x in self.label_names}
        # ... store the number of image channels
        self.nchannels_image = self.array_dict[list(self.array_dict.keys())[0]].shape[self.channel_info_image.index('c')] if 'c' in self.channel_info_image else None
        # ... store channel annotation from OMERO
        self.channels = []
        if 'omero' in self.zarr_group.attrs and 'channels' in self.zarr_group.attrs['omero']:
            self.channels = self.zarr_group.attrs['omero']['channels']
        elif self.nchannels_image is not None:
            self.channels = [{'label': f'channel-{i+1}',
                              'color': '00FFFF'} for i in range(self.nchannels_image)]
       
    @staticmethod
    def _load_multiscale_info(group: zarr.Group,
                              skip_checks: Optional[bool]=False) -> dict[str, Any]:
        if 'multiscales' not in group.attrs:
            raise ValueError(f"no multiscale info found in {group.path}")
        if len(group.attrs['multiscales']) > 1:
            warnings.warn(f"{group.path} contains more than one multiscale - using the first one")
        info = group.attrs['multiscales'][0]
        if not skip_checks:
            if 'axes' not in info:
                raise ValueError(f"no axes info found in 'multiscales' of {group.path}")
            # TODO: add further checks
        return info

    @staticmethod
    def _load_axes_unit(multiscale_dict: dict[str, Any]) -> str:
        supported_units = ['micrometer', 'pixel']
        unit_set = set([x['unit'] for x in multiscale_dict['axes'] if x['type'] == 'space' and 'unit' in x])
        if len(unit_set) == 0 or (len(unit_set) == 1 and 'unit' in unit_set):
            return 'pixel'
        elif len(unit_set) == 1 and list(unit_set)[0] in supported_units:
            return list(unit_set)[0]
        else:
            raise ValueError(f"unsupported unit in multiscale_info: {multiscale_dict}")
    
    @staticmethod
    def _load_channel_info(multiscale_dict: dict[str, Any]) -> str:
        type2ch = {'time': 't', 'channel': 'c', 'space': 'S'}
        spatial = ['x', 'y', 'z']
        L = [type2ch[d['type']] for d in multiscale_dict['axes']]
        L.reverse()
        while 'S' in L:
            L[L.index('S')] = spatial.pop(0)
        L.reverse()
        return ''.join(L)
            
    
    @staticmethod
    def _extract_scale_spacings(dataset_dict: dict[str, Any]) -> list[str, list[float]]:
        if 'path' not in dataset_dict or 'coordinateTransformations' not in dataset_dict or 'scale' not in dataset_dict['coordinateTransformations'][0]:
            raise ValueError("could not extract zyx spacing from multiscale_info")
        return [dataset_dict['path'], dataset_dict['coordinateTransformations'][0]['scale']]
    
    @staticmethod
    def _find_path_of_lowest_level(datasets: list[dict[str, Any]]) -> str:
        lev = None
        maxx = 0 # maximal x resolution
        for i in range(len(datasets)):
            if datasets[i]['coordinateTransformations'][0]['scale'][-1] > maxx:
                lev = str(datasets[i]['path'])
                maxx = datasets[i]['coordinateTransformations'][0]['scale'][-1]
        return lev

    def _digest_pyramid_level_argument(self, pyramid_level=None, label_name=None) -> str:
        """
        [internal] Interpret a `pyramid_level` argument in the context of a given Image object.

        Parameters:
            pyramid_level (int, str or None): pyramid level, coerced to str. If None,
                the last pyramid level (typically the lowest-resolution one) will be returned.
            label_name (str or None): defines what `pyramid_level` refers to. If None,
                it refers to the intensity image. Otherwise, it refers to a label with
                the name given by `label_name`. For example, to select the 'nuclei' labels,
                the argument would be set to `nuclei`.

        Returns:
            Integer index of the pyramid level.
        """
        if label_name is not None and label_name not in self.label_names:
            raise ValueError(f"invalid label name '{label_name}' - must be one of {self.label_names}")
        if pyramid_level == None: 
            # no pyramid level given -> pick lowest resolution one
            if label_name is None: # intensity image
                pyramid_level = self._find_path_of_lowest_level(self.multiscales_image['datasets'])
            else: # label image
                pyramid_level = self._find_path_of_lowest_level(self.multiscales_labels[label_name]['datasets'])
        else:
            # make sure it is a string
            pyramid_level = str(pyramid_level)
            # make sure it exists
            if label_name is None: # intensity image
                pyramid_level_names = [str(x['path']) for x in self.multiscales_image['datasets']]
            else: # label image
                pyramid_level_names = [str(x['path']) for x in self.multiscales_labels[label_name]['datasets']]
            if pyramid_level not in pyramid_level_names:
                raise ValueError(f"invalid pyramid level '{pyramid_level}' - must be one of {pyramid_level_names}")
        return pyramid_level
    
    # string representation ---------------------------------------------------
    def __str__(self):
        nch = self.nchannels_image
        chlabs = ', '.join([x['label'] for x in self.channels])
        npl = len(self.multiscales_image['datasets'])
        segnames = ', '.join(self.label_names)
        tabnames = ', '.join(self.table_names)
        spatial_dims = [i for i in range(len(self.channel_info_image)) if self.channel_info_image[i] in ['z', 'y', 'x']]
        zyx = ''.join([self.channel_info_image[i] for i in spatial_dims])
        image_spacings = {x[0]: [x[1][s] for s in spatial_dims] for x in [self._extract_scale_spacings(y) for y in self.multiscales_image['datasets']]}
        pl_scalefactor = 'None (only one pyramid level)'
        if len(image_spacings) > 1:
            pl_nms = list(image_spacings.keys())
            pl_scalefactor = np.divide(image_spacings[pl_nms[1]], image_spacings[pl_nms[0]])
        return f"Image {self.name}\n  path: {self.path}\n  n_channels: {nch} ({chlabs})\n  n_pyramid_levels: {npl}\n  pyramid_{zyx}_scalefactor: {pl_scalefactor}\n  full_resolution_{zyx}_spacing ({self.axes_unit_image}): {image_spacings[list(image_spacings.keys())[0]]}\n  segmentations: {segnames}\n  tables (measurements): {tabnames}\n"
    
    def __repr__(self):
        return str(self)
    
    # coordinate conversion ---------------------------------------------------
    ## REMARK: could be moved to utils.py
    @staticmethod
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

            >>> y, x = img.convert_coordinates((10,30), [1, 1], [2, 3])
        """
        # digest arguments
        assert len(coords_from) == len(scale_from)
        assert len(coords_from) == len(scale_to)

        # convert and return
        coords_to = np.divide(np.multiply(coords_from, scale_from), scale_to)
        return(tuple(coords_to))

    # accessor methods -----------------------------------------------------------
    def get_scale(self,
                  pyramid_level: str,
                  label_name: Optional[str]=None) -> list[float]:
        """
        Get the scale of a given pyramid level.

        Parameters:
            pyramid_level (str): The pyramid level from which to get the scale.
            label_name (str or None): The name of the label image to which
                `pyramid_level` refers to. If None, `pyramid_level` is assumed
                to refer to the intensity image.
        
        Returns:
            A list with the scale of the given pyramid level.
        
        Example:
            Get the scale of the first pyramid level:

            >>> scale = img.get_scale('level_0', 'nuclei')
        """
        # digest arguments
        pyramid_level = self._digest_pyramid_level_argument(pyramid_level, label_name)

        # extract scale
        if label_name:
            datasets_list = self.multiscales_labels[label_name]['datasets']
        else:
            datasets_list = self.multiscales_image['datasets']
        scale = [x['coordinateTransformations'][0]['scale'] for x in datasets_list if str(x['path']) == pyramid_level][0]

        # return
        return scale
    
    def get_array_by_coordinate(self,
                                label_name: Optional[str]=None,
                                upper_left_yx: Optional[tuple[int]]=None,
                                lower_right_yx: Optional[tuple[int]]=None,
                                size_yx: Optional[tuple[int]]=None,
                                coordinate_unit: str='micrometer',
                                pyramid_level: Optional[str]=None,
                                pyramid_level_coord: Optional[str]=None,
                                as_NumPy: bool=False) -> Union[dask.array.Array, np.ndarray]:
        """
        Extract a (sub)array from an image (intensity image or label) by coordinates.

        None or exactly two of `upper_left_yx`, `lower_right_yx` and `size_yx`
        need to be given. If none are given, it will return the full image.
        Otherwise, `upper_left_yx` contains the lower indices than `lower_right_yx`
        (origin on the top-left, zero-based coordinates), and each of them is
        a tuple of (y, x). No t, c or z coordinate need to be given, all of them
        are returned if there are several ones.

        Parameters:
            label_name (str or None): The name of the label image to be extracted.
                If `None`, the intensity image will be extracted.
            upper_left_yx (tuple): Tuple of (y, x) coordinates for the upper-left
                (lower) coordinates defining the region of interest.
            lower_right_yx (tuple): Tuple of (y, x) coordinates for the lower-right
                (higher) coordinates defining the region of interest.
            size_yx (tuple): Tuple of (size_y, size_x) defining the size of the
                region of interest.
            coordinate_unit (str): The unit of the image coordinates, for example
                'micrometer' or 'pixel'.
            pyramid_level (str): The pyramid level (resolution level), from which the
                array should be extracted. If `None`, the lowest-resolution
                pyramid level will be selected.
            pyramid_level_coord (str): An optional integer scalar giving the 
                image pyramid level to which the coordinates (`upper_left_yx`,
                `lower_right_yx` and `size_yx`) refer to if `coordinate_unit="pixel"`
                (it is ignored otherwise). By default, this is `None`, which will
                use `pyramid_level`.
            as_NumPy (bool): If `True`, return the image as a `numpy.ndarray`
                object (e.g. c,z,y,x). Otherwise, return the (on-disk) `dask`
                array of the same dimensions.
        
        Returns:
            The extracted array, either as a `dask.array.Array` on-disk array,
            or as an in-memory `numpy.ndarray` if `as_NumPy=True`.
        
        Examples:
            Obtain the whole image of the lowest-resolution as an array:

            >>> img.get_array_by_coordinate()
        """
        # digest arguments
        pyramid_level = self._digest_pyramid_level_argument(pyramid_level, label_name)

        # load image
        if label_name:
            arr = self.zarr_group['labels'][label_name][pyramid_level]
        else:
            arr = self.zarr_group[pyramid_level]

        # calculate corner coordinates and subset if needed
        num_unknowns = sum([x == None for x in [upper_left_yx, lower_right_yx, size_yx]])
        if num_unknowns == 1:
            if size_yx:
                assert all([x > 0 for x in size_yx]), 'size_yx values need to be positive'
                if not upper_left_yx:
                    upper_left_yx = tuple(lower_right_yx[i] - size_yx[i] for i in range(2))
                elif not lower_right_yx:
                    lower_right_yx = tuple(upper_left_yx[i] + size_yx[i] for i in range(2))
            assert all([upper_left_yx[i] < lower_right_yx[i] for i in range(len(upper_left_yx))]), 'upper_left_yx needs to be less than lower_right_yx'

            # convert coordinates if needed
            if coordinate_unit != "pixel" or (pyramid_level != pyramid_level_coord):
                if coordinate_unit == "micrometer":
                    # Note: this assumes that all non-spatial dimensions
                    #       (channels, time) have scales of 1.0
                    scale_from = [1.0] * len(arr.shape)
                elif coordinate_unit == "pixel":
                    scale_from = self.get_scale(pyramid_level_coord or pyramid_level, label_name)
                else:
                    raise ValueError("`coordinate_unit` needs to be 'micrometer' or 'pixel'")
                scale_to = self.get_scale(pyramid_level, label_name)

                upper_left_yx = self.convert_coordinates(upper_left_yx, scale_from[-2:], scale_to[-2:])
                lower_right_yx = self.convert_coordinates(lower_right_yx, scale_from[-2:], scale_to[-2:])

            # slice array
            arr = arr[...,
                      slice(int(upper_left_yx[0]), int(lower_right_yx[0]) + 1),
                      slice(int(upper_left_yx[1]), int(lower_right_yx[1]) + 1)]
        elif num_unknowns != 3:
            raise ValueError("Either none or two of `upper_left_yx`, `lower_right_yx` and `size_yx` have to be given")

        # convert if needed and return
        if as_NumPy:
            arr = np.array(arr)
        return arr

    # plotting methods -----------------------------------------------------------
    def plot(self,
             upper_left_yx: Optional[tuple[int]]=None,
             lower_right_yx: Optional[tuple[int]]=None,
             size_yx: Optional[tuple[int]]=None,
             coordinate_unit: str='micrometer',
             label_name: Optional[str]=None,
             pyramid_level: Optional[str]=None,
             pyramid_level_coord: Optional[str]=None,
             channels_labels: Optional[list[str]]=None,
             scalebar_micrometer: int=0,
             show_scalebar_label: [bool]=True,
             **kwargs: Any) -> None:
        """
        Plot an image.
         
        Plot a single image or a part of it, optionally selecting
        channels by label and extracting colors from the OMERO metadata
        using data at resolution `pyramid_level`.

        Parameters:
            upper_left_yx (tuple): Tuple of (y, x) coordinates for the upper-left
                (lower) coordinates defining the region of interest.
            lower_right_yx (tuple): Tuple of (y, x) coordinates for the lower-right
                (higher) coordinates defining the region of interest.
            size_yx (tuple): Tuple of (size_y, size_x) defining the size of the
                region of interest.
            coordinate_unit (str): The unit of the image coordinates, for example
                'micrometer' or 'pixel'.
            label_name (str): The name of the a segmentation mask to be plotted
                semi-transparently over the image. If `None`, just the image
                is plotted.
            pyramid_level (str): The pyramid level (resolution level), from
                which the image should be extracted. If `None`, the
                lowest-resolution (highest) pyramid level will be selected.
            pyramid_level_coord (str): An optional integer scalar giving the 
                image pyramid level to which the coordinates (`upper_left_yx`,
                `lower_right_yx` and `size_yx`) refer to if `coordinate_unit="pixel"`
                (it is ignored otherwise). By default, this is `None`, which will
                use `pyramid_level`.
            channels_labels (list[str]): The labels of the image channel(s) to
                be plotted. This provides an alternative to selecting `channels`
                by index.
            scalebar_micrometer (int): If non-zero, add a scale bar corresponding
                to `scalebar_micrometer` to the image.
            show_scalebar_label (bool):  If `True`, add micrometer label to scale bar.
            **kwargs: Additional arguments for `plotting.plot_image`, for example
                'channels', 'channel_colors', 'channel_ranges', 'z_projection_method',
                etc. For a full list of available arguments, see
                [plotting.plot_image documentation](plotting.md#src.ez_zarr.plotting.plot_image).

        Examples:
            Plot the whole image `img`.

            >>> img.plot(channels = [0], channel_colors = ['red'])
        """
        # digest arguments
        assert label_name == None or label_name in self.label_names, (
            f"Unknown label_name ({label_name}), should be `None` or one of "
            ', '.join(self.label_names)
        )
        img_pl = self._digest_pyramid_level_argument(pyramid_level=pyramid_level,
                                                     label_name=None)
        # import optional modules
        plotting = importlib.import_module('ez_zarr.plotting')

        # get channel indices
        if channels_labels != None:
            if 'channels' in kwargs:
                raise Warning('`channels` will be ignored if `channels_labels` is given')
            all_channels_labels = [ch['label'] for ch in self.channels]
            missing_labels = [channels_labels[i] for i in range(len(channels_labels)) if channels_labels[i] not in all_channels_labels]
            if len(missing_labels) > 0:
                raise ValueError(f"Unknown channels_labels ({', '.join(missing_labels)}), should be `None` or one of {', '.join(all_channels_labels)}")
            kwargs['channels'] = [all_channels_labels.index(x) for x in channels_labels]
        
        # extract `channel_colors`
        if 'channels' in kwargs and ('channel_colors' not in kwargs or  len(kwargs['channels']) != len(kwargs['channel_colors'])):
            print("extracting `channel_colors` from image metadata")
            kwargs['channel_colors'] = ['#' + self.channels[i]['color'] for i in kwargs['channels']]
        
        # extract `channel_ranges`
        if 'channels' in kwargs and ('channel_ranges' not in kwargs or len(kwargs['channels']) != len(kwargs['channel_ranges'])):
            print("setting `channel_ranges` based on length of `channels`")
            kwargs['channel_ranges'] = [[0.01, 0.95] for i in range(len(kwargs['channels']))]

        # extract scale information for image
        scale_img = [x['coordinateTransformations'][0]['scale'] for x in self.multiscales_image['datasets'] if str(x['path']) == img_pl][0]
        scale_img_spatial = [scale_img[i] for i in range(len(scale_img)) if self.channel_info_image[i] in ['z', 'y', 'x']]

        # get label pyramid level closest to `img_pl`
        if label_name != None:
            # extract scale information for labels
            scale_lab = [self._extract_scale_spacings(x) for x in self.multiscales_labels[label_name]['datasets']]
            scale_lab_spatial = [[s[0], [s[1][i] for i in range(len(s[1])) if self.channel_info_labels[label_name][i] in ['z', 'y', 'x']]] for s in scale_lab]
            # find nearest scale
            nearest_scale_idx = np.argmin([np.linalg.norm(np.array(s[1]) - np.array(scale_img_spatial)) for s in scale_lab_spatial])
            label_pl = scale_lab[nearest_scale_idx][0]

            # TODO: automatically upscale labels if necessary
            if np.any(scale_img_spatial != scale_lab_spatial[nearest_scale_idx][1]):
                raise NotImplementedError('Automatic upscaling of labels not yet implemented')

        # get well image
        img = self.get_array_by_coordinate(label_name=None,
                                           upper_left_yx=upper_left_yx,
                                           lower_right_yx=lower_right_yx,
                                           size_yx=size_yx,
                                           coordinate_unit=coordinate_unit,
                                           pyramid_level=img_pl,
                                           pyramid_level_coord=pyramid_level_coord,
                                           as_NumPy=True)

        if label_name != None:
            lab = self.get_array_by_coordinate(label_name=label_name,
                                               upper_left_yx=upper_left_yx,
                                               lower_right_yx=lower_right_yx,
                                               size_yx=size_yx,
                                               coordinate_unit=coordinate_unit,
                                               pyramid_level=label_pl,
                                               pyramid_level_coord=pyramid_level_coord,
                                               as_NumPy=True)
            assert img.shape[1:] == lab.shape, (
                f"label {label_name} shape {lab.shape} does not match "
                f"image shape {img.shape}"
            )
        else:
            lab = None

        # calculate scalebar length in pixel in x direction
        if scalebar_micrometer != 0:
            kwargs['scalebar_pixel'] = self.convert_coordinates(
                coords_from = (scalebar_micrometer,),
                scale_from = [1.0],
                scale_to = [self.get_scale(pyramid_level=img_pl, label_name=None)[-1]])[0]
        else:
            kwargs['scalebar_pixel'] = 0

        # plot image
        if show_scalebar_label:
            kwargs['scalebar_label'] = str(scalebar_micrometer) + ' µm'
        else:
            kwargs['scalebar_label'] = None
        kwargs['im'] = img
        kwargs['msk'] = lab
        kwargs['spacing_yx'] = scale_img_spatial[-2:]
        plotting.plot_image(**kwargs)

