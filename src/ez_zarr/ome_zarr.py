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
from ez_zarr.utils import convert_coordinates, resize_image


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
    def _find_path_of_lowest_resolution_level(datasets: list[dict[str, Any]]) -> str:
        lev = None
        maxx = 0 # maximal x pixel size (lowest resolution)
        for i in range(len(datasets)):
            if datasets[i]['coordinateTransformations'][0]['scale'][-1] > maxx:
                lev = str(datasets[i]['path'])
                maxx = datasets[i]['coordinateTransformations'][0]['scale'][-1]
        return lev

    @staticmethod
    def _find_path_of_highest_resolution_level(datasets: list[dict[str, Any]]) -> str:
        lev = None
        minx = float('inf') # minimal x pixel size (highest resolution)
        for i in range(len(datasets)):
            if datasets[i]['coordinateTransformations'][0]['scale'][-1] < minx:
                lev = str(datasets[i]['path'])
                minx = datasets[i]['coordinateTransformations'][0]['scale'][-1]
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
                pyramid_level = self._find_path_of_lowest_resolution_level(self.multiscales_image['datasets'])
            else: # label image
                pyramid_level = self._find_path_of_lowest_resolution_level(self.multiscales_labels[label_name]['datasets'])
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
    
    def _digest_channels_labels(self, channels_labels: list[str]) -> list[int]:
            all_channels_labels = [ch['label'] for ch in self.channels]
            missing_labels = [channels_labels[i] for i in range(len(channels_labels)) if channels_labels[i] not in all_channels_labels]
            if len(missing_labels) > 0:
                raise ValueError(f"Unknown channels_labels ({', '.join(missing_labels)}), should be `None` or one of {', '.join(all_channels_labels)}")
            return [all_channels_labels.index(x) for x in channels_labels]
    
    def _digest_bounding_box(self,
                             upper_left_yx: Optional[tuple[int]]=None,
                             lower_right_yx: Optional[tuple[int]]=None,
                             size_yx: Optional[tuple[int]]=None,
                             coordinate_unit: str='micrometer',
                             label_name: Optional[str]=None,
                             pyramid_level: Optional[str]=None,
                             pyramid_level_coord: Optional[str]=None) -> list[tuple[int]]:
        """
        Solves for the bounding box defined by upper-left, lower-right or size.

        None or exactly two of `upper_left_yx`, `lower_right_yx` and `size_yx`
        need to be given. If none are given, the bounding box will correspond to
        the full image.
        Otherwise, `upper_left_yx` contains lower values than `lower_right_yx`
        (origin on the top-left, zero-based coordinates), and each of them is
        a tuple of (y, x).
        If `coordinate_unit` is not 'pixel', or `coordinate_unit` is 'pixel' and
        `pyramid_level_coord` is not None and different from `pyramid_level`, the
        solved bounding box coordinates are converted to pixels in `pyramid_level`.

        Parameters:
            upper_left_yx (tuple, optional): Tuple of (y, x) coordinates for the upper-left
                (lower) coordinates defining the region of interest.
            lower_right_yx (tuple, optional): Tuple of (y, x) coordinates for the lower-right
                (higher) coordinates defining the region of interest.
            size_yx (tuple, optional): Tuple of (size_y, size_x) defining the size of the
                region of interest.
            coordinate_unit (str): The unit of the image coordinates, for example
                'micrometer' or 'pixel'.
            label_name (str, optional): The name of the label image that the coordinates
                refer to. If `None`, the intensity image will be used as reference.
            pyramid_level (str): The pyramid level (resolution level), to which the
                returned coordinates should refer. If `None`, the lowest-resolution
                pyramid level will be selected.
            pyramid_level_coord (str, optional): An optional string giving the 
                image pyramid level to which the input coordinates (`upper_left_yx`,
                `lower_right_yx` and `size_yx`) refer to if `coordinate_unit="pixel"`
                (it is ignored otherwise). By default, this is `None`, which will
                use `pyramid_level`.
        
        Returns:
            A list of upper-left and lower-right tuple coordinates for the bounding box: `[(y1, x1), (y2, x2)]`. These coordinates
            are always in 'pixel' units of `pyramid_level`.
        
        Examples:
            Obtain the whole coordinates of the full image at pyramid level 0:

            >>> img._digest_bounding_box(pyramid_level=0)
        """
        # digest arguments
        pyramid_level = self._digest_pyramid_level_argument(pyramid_level, label_name)

        # get image array
        if label_name:
            arr = self.zarr_group['labels'][label_name][pyramid_level]
        else:
            arr = self.zarr_group[pyramid_level]

        # calculate corner coordinates
        num_unknowns = sum([x == None for x in [upper_left_yx, lower_right_yx, size_yx]])
        if num_unknowns == 1:
            if size_yx:
                assert all([x > 0 for x in size_yx]), 'size_yx values need to be positive'
                if not upper_left_yx:
                    upper_left_yx = tuple(lower_right_yx[i] - size_yx[i] for i in range(2))
                elif not lower_right_yx:
                    lower_right_yx = tuple(upper_left_yx[i] + size_yx[i] for i in range(2))
            assert all([upper_left_yx[i] < lower_right_yx[i] for i in range(len(upper_left_yx))]), 'upper_left_yx needs to be less than lower_right_yx'
        elif num_unknowns == 3:
            upper_left_yx = (0, 0)
            lower_right_yx = (arr.shape[-2], arr.shape[-1])
            coordinate_unit = 'pixel'
            pyramid_level_coord = pyramid_level
        else:
            raise ValueError("Either none or two of `upper_left_yx`, `lower_right_yx` and `size_yx` have to be given")

        # convert coordinates if needed
        if coordinate_unit != "pixel" or (pyramid_level != pyramid_level_coord):
            if coordinate_unit == "micrometer":
                # Note: this assumes that all non-spatial dimensions
                #       (channels, time) have scales of 1.0
                scale_from = [1.0] * len(arr.shape)
            elif coordinate_unit == "pixel":
                if pyramid_level_coord is None:
                    pyramid_level_coord = pyramid_level
                else:
                    pyramid_level_coord = self._digest_pyramid_level_argument(pyramid_level_coord, label_name)
                scale_from = self.get_scale(pyramid_level_coord, label_name)
            else:
                raise ValueError("`coordinate_unit` needs to be 'micrometer' or 'pixel'")
            scale_to = self.get_scale(pyramid_level, label_name)

            upper_left_yx = convert_coordinates(upper_left_yx, scale_from[-2:], scale_to[-2:])
            lower_right_yx = convert_coordinates(lower_right_yx, scale_from[-2:], scale_to[-2:])

            # convert to int
            upper_left_yx = tuple(int(round(x)) for x in upper_left_yx)
            lower_right_yx = tuple(int(round(x)) for x in lower_right_yx)

        # return
        return([upper_left_yx, lower_right_yx])
    
    def _get_bounding_box_for_label_value(self,
                                          label_name: str,
                                          label_value: Union[int, float, str],
                                          label_pyramid_level: Union[int, str],
                                          padding_pixels: Optional[int]=0) -> tuple[tuple[int], tuple[int]]:
        """
        Given a label name and value, find the corner coordinates of the bounding box.

        Parameters:
            label_name (str): The name of the label image to extract the bounding box from.
            label_value (int, float, str): The value of the label to extract the bounding box for.
            label_pyramid_level (int, str): The pyramid level to extract the bounding box from.
            padding_pixels (int): The number of pixels to add to each side for each axis of the bounding box.
        
        Returns:
            A tuple of upper-left and lower-right pixel coordinates for the bounding box
            containing the label value, or `(None, None)` if the label value is not found.
            For a 2D label image, this would be `((y1, x1), (y2, x2))`.
        
        Example:
            Get the bounding box for the label value 1 in pyramid level 0:

            >>> img._get_bounding_box_for_label_value(label_name='nuclei', label_value=1, label_pyramid_level=0)
        """
        # digest arguments
        assert label_name in self.label_names, f'`label_name` must be in {self.label_names}'
        label_pyramid_level = self._digest_pyramid_level_argument(label_pyramid_level, label_name)
        assert isinstance(padding_pixels, int), '`padding_pixels` must be an integer scalar'

        # get label array
        lab_arr = self.get_array_by_coordinate(label_name=label_name,
                                               pyramid_level=label_pyramid_level)

        # find bounding box
        value_coordinates = np.equal(lab_arr, label_value).nonzero()
        if len(value_coordinates[0]) == 0:
            return tuple([None, None])
        upper_left = tuple([min(x) for x in value_coordinates])
        lower_right = tuple([max(x) + 1 for x in value_coordinates])

        # padding
        upper_left = tuple([max(0, upper_left[i] - padding_pixels) for i in range(len(upper_left))])
        lower_right = tuple([min(lab_arr.shape[i], lower_right[i] + padding_pixels) for i in range(len(lower_right))])

        # return result
        return tuple([upper_left, lower_right])
    
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
    
    # accessor methods -----------------------------------------------------------
    def get_path(self) -> str:
        """Get the path of an OME-Zarr image.
        
        Returns:
            The path to the OME-Zarr image.
        """
        return self.path

    def get_channels(self) -> list:
        """Get info on channels in an OME-Zarr image.
        
        Returns:
            A list of dicts with information on channels.
        """
        return self.channels
    
    def get_label_names(self) -> list:
        """Get list of label names in an OME-Zarr image.
        
        Returns:
            A list of label names (str) available for the image.
        """
        return self.label_names

    def get_table_names(self) -> list:
        """Get list of table names in an OME-Zarr image.
        
        Returns:
            A list of table names (str) available for the image.
        """
        return self.table_names
    
    def get_pyramid_levels(self, label_name: Optional[str]=None) -> list[str]:
        """Get list of pyramid levels in an OME-Zarr image.
        
        Parameters:
            label_name (str, optional): The name of the label image for which
                to return pyramid levels. If None, pyramid levels from the
                intensity image are returned.
        
        Returns:
            A list of available pyramid levels (str).
        """
        if label_name:
            return [str(x['path']) for x in self.multiscales_labels[label_name]['datasets']]
        else:
            return [str(x['path']) for x in self.multiscales_image['datasets']]

    def get_scale(self,
                  pyramid_level: str,
                  label_name: Optional[str]=None,
                  spatial_axes_only: bool=False) -> list[float]:
        """
        Get the scale of a given pyramid level.

        Parameters:
            pyramid_level (str): The pyramid level from which to get the scale.
            label_name (str or None): The name of the label image to which
                `pyramid_level` refers to. If None, `pyramid_level` is assumed
                to refer to the intensity image.
            spatial_axes_only (bool): If True, only the scales for spatial
                dimensions are returned.
        
        Returns:
            A list with the axis scales of the given pyramid level.
        
        Example:
            Get the axis scales of the first pyramid level:

            >>> scale = img.get_scale('level_0', 'nuclei')
        """
        # digest arguments
        pyramid_level = self._digest_pyramid_level_argument(pyramid_level, label_name)

        # extract scale
        if label_name:
            datasets_list = self.multiscales_labels[label_name]['datasets']
            channel_info = self.channel_info_labels[label_name]
        else:
            datasets_list = self.multiscales_image['datasets']
            channel_info = self.channel_info_image
        scale = [x['coordinateTransformations'][0]['scale']
                 for x in datasets_list
                 if str(x['path']) == pyramid_level][0]

        if spatial_axes_only:
            scale = [scale[i]
                     for i in range(len(scale))
                     if channel_info[i] in ['z', 'y', 'x']]
        
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
        need to be given. If none are given, the full image is returned.
        Otherwise, `upper_left_yx` contains the lower indices than `lower_right_yx`
        (origin on the top-left, zero-based coordinates), and each of them is
        a tuple of (y, x). No t, c or z coordinate need to be given, all of them
        are returned if there are several ones.

        Parameters:
            label_name (str, optional): The name of the label image to be extracted.
                If `None`, the intensity image will be extracted.
            upper_left_yx (tuple, optional): Tuple of (y, x) coordinates for the upper-left
                (lower) coordinates defining the region of interest.
            lower_right_yx (tuple, optional): Tuple of (y, x) coordinates for the lower-right
                (higher) coordinates defining the region of interest.
            size_yx (tuple, optional): Tuple of (size_y, size_x) defining the size of the
                region of interest.
            coordinate_unit (str): The unit of the image coordinates, for example
                'micrometer' or 'pixel'.
            pyramid_level (str): The pyramid level (resolution level), from which the
                array should be extracted. If `None`, the lowest-resolution
                pyramid level will be selected.
            pyramid_level_coord (str, optional): An optional string giving the 
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

        # calculate corner coordinates
        upper_left_yx, lower_right_yx = self._digest_bounding_box(
            upper_left_yx=upper_left_yx,
            lower_right_yx=lower_right_yx,
            size_yx=size_yx,
            coordinate_unit=coordinate_unit,
            label_name=label_name,
            pyramid_level=pyramid_level,
            pyramid_level_coord=pyramid_level_coord)
        
        # subset array if needed
        if upper_left_yx != (0, 0) or lower_right_yx != arr.shape[-2:]:
            arr = arr[...,
                      slice(upper_left_yx[0], lower_right_yx[0]),
                      slice(upper_left_yx[1], lower_right_yx[1])]

        # convert if needed and return
        if as_NumPy:
            arr = np.array(arr)
        return arr

    def get_array_pair_by_coordinate(self,
                                     label_name: str,
                                     upper_left_yx: Optional[tuple[int]]=None,
                                     lower_right_yx: Optional[tuple[int]]=None,
                                     size_yx: Optional[tuple[int]]=None,
                                     coordinate_unit: str='micrometer',
                                     pyramid_level: Optional[str]=None,
                                     pyramid_level_coord: Optional[str]=None) -> tuple[np.ndarray]:
        """
        Extract a matching pair of (sub)arrays (intensity and label) by coordinates.

        None or exactly two of `upper_left_yx`, `lower_right_yx` and `size_yx`
        need to be given. If none are given, the full image is returned.
        Otherwise, `upper_left_yx` contains the lower indices than `lower_right_yx`
        (origin on the top-left, zero-based coordinates), and each of them is
        a tuple of (y, x). No t, c or z coordinate need to be given, all of them
        are returned if there are several ones.

        Coordinate and pyramid level arguments all refer to the intensity image.
        For the label, matching values will be selected automatically, and if
        necessary, the label array is resized to match the intensity array.

        Parameters:
            label_name (str): The name of the label image to be extracted.
            upper_left_yx (tuple, optional): Tuple of (y, x) intensity image
                coordinates for the upper-left (lower) coordinates defining the
                region of interest.
            lower_right_yx (tuple, optional): Tuple of (y, x) intensity image
                coordinates for the lower-right (higher) coordinates defining the
                region of interest.
            size_yx (tuple, optional): Tuple of (size_y, size_x) defining the size
                of the intensity image region of interest.
            coordinate_unit (str): The unit of the image coordinates, for example
                'micrometer' or 'pixel'.
            pyramid_level (str): The intensity image pyramid level (resolution
                level), from which the intensity array should be extracted.
                If `None`, the lowest-resolution pyramid level will be selected.
                A matching pyramid level for the label will be selected
                automatically.
            pyramid_level_coord (str, optional): An optional string giving the 
                intensity image pyramid level to which the coordinates (if any)
                refer to if `coordinate_unit="pixel"` (it is ignored otherwise).
                By default, this is `None`, which will use `pyramid_level`.
        
        Returns:
            A tuple of two `numpy.ndarray` objects with the extracted intensity
            and (possibly resized) label arrays.
        
        Examples:
            Obtain the whole image and matching 'organoids' label arrays:

            >>> img, lab = img.get_array_pair_by_coordinate(label_name = 'organoids')
        """
        # digest arguments
        assert isinstance(label_name, str) and label_name in self.label_names, (
            f"Unknown label_name ({label_name}), should be one of "
            ', '.join(self.label_names)
        )
        pyramid_level = self._digest_pyramid_level_argument(
            pyramid_level=pyramid_level,
            label_name=None
        )
        if pyramid_level_coord is None:
            pyramid_level_coord = pyramid_level

        # find matching label pyramid level
        img_scale_spatial = self.get_scale(
            pyramid_level=pyramid_level,
            label_name=None,
            spatial_axes_only=True
        )
        lab_scale_spatial_dict = {
            pl: self.get_scale(pyramid_level=pl, label_name=label_name, spatial_axes_only=True) for pl in self.get_pyramid_levels(label_name=label_name)
        }
        # ... filter out label scales with higher resolution than the intensity image
        lab_scale_spatial_dict = {pl: lab_scale_spatial for pl, lab_scale_spatial in lab_scale_spatial_dict.items() if all([lab_scale_spatial[i] >= img_scale_spatial[i] for i in range(len(lab_scale_spatial))])}
        if len(lab_scale_spatial_dict) == 0:
            raise ValueError(f"For the requested pyramid level ({pyramid_level}) of the intensity image, down-scaling of an available label ('{label_name}') would be required. Down-scaling of labels is not supported - try selecting a higher-resolution intensity image.")

        nearest_scale_idx = np.argmin([np.mean(np.array(lab_scale_spatial_dict[pl]) / np.array(img_scale_spatial)) for pl in lab_scale_spatial_dict.keys()])
        nearest_scale_pl = list(lab_scale_spatial_dict.keys())[nearest_scale_idx]
        lab_scale_spatial = lab_scale_spatial_dict[nearest_scale_pl]

        # calculate image corner points
        imgpixel_upper_left_yx, imgpixel_lower_right_yx = self._digest_bounding_box(
            upper_left_yx=upper_left_yx,
            lower_right_yx=lower_right_yx,
            size_yx=size_yx,
            coordinate_unit=coordinate_unit,
            label_name=None,
            pyramid_level=pyramid_level,
            pyramid_level_coord=pyramid_level_coord
        )

        # make sure that the dimensions are divisible by
        # the yx scaling factor between intensity and label arrays
        scalefact_yx = np.divide(lab_scale_spatial, img_scale_spatial)
        imgpixel_upper_left_yx = tuple((np.floor_divide(imgpixel_upper_left_yx, scalefact_yx[-2:]) * scalefact_yx[-2:]))
        imgpixel_lower_right_yx = tuple((np.floor_divide(imgpixel_lower_right_yx, scalefact_yx[-2:]) * scalefact_yx[-2:]))

        # get intensity array
        img_arr = np.array(self.get_array_by_coordinate(
            upper_left_yx=imgpixel_upper_left_yx,
            lower_right_yx=imgpixel_lower_right_yx,
            size_yx=None,
            coordinate_unit='pixel',
            label_name=None,
            pyramid_level=pyramid_level
        ))

        # convert intensity coordiantes to label coordinates
        labpixel_upper_left_yx = convert_coordinates(
            imgpixel_upper_left_yx,
            img_scale_spatial[-2:],
            lab_scale_spatial[-2:]
        )
        labpixel_lower_right_yx = convert_coordinates(
            imgpixel_lower_right_yx,
            img_scale_spatial[-2:],
            lab_scale_spatial[-2:]
        )

        # get label array
        lab_arr = np.array(self.get_array_by_coordinate(
            upper_left_yx=labpixel_upper_left_yx,
            lower_right_yx=labpixel_lower_right_yx,
            size_yx=None,
            coordinate_unit='pixel',
            label_name=label_name,
            pyramid_level=nearest_scale_pl
        ))

        # resize label if needed (correct non-matching scales or rounding errors)
        if lab_arr.shape[-2:] != img_arr.shape[-2:]:
            warnings.warn(f"For the requested pyramid level ({pyramid_level}) of the intensity image, no matching label ('{label_name}') is available. Up-scaling the label using factor(s) {scalefact_yx}")
            lab_arr = resize_image(
                im=lab_arr,
                output_shape=img_arr.shape[(img_arr.ndim-lab_arr.ndim):],
                im_type='label',
                number_nonspatial_axes=sum([int(s not in ['z','y','x']) for s in self.channel_info_labels[label_name]])
            )

        # return arrays
        return tuple([img_arr, lab_arr])


    def get_table(self,
                  table_name: str,
                  as_AnnData: bool=False) -> Any:
        """Extract tabular data (for example quantifications) for OME-Zarr image.
        
        Parameters:
            table_name (str): The name of the table to extract.
            as_AnnData (bool): If `True`, the table is returned as an `AnnData` object, otherwise it is returned as a `pandas.DataFrame`.
        
        Returns:
            The extracted table, either as an `anndata.AnnData` object if `as_AnnData=True`, or a `pandas.DataFrame` otherwise. `None` if `table_name` is not found.
        
        Examples:
            List available tables names, and get table 'FOV_ROI_table':

            >>> img.get_table_names()
            >>> img.get_table(table_name='FOV_ROI_table')
        """

        # check for existing table_name
        if table_name not in self.table_names:
            warnings.warn(f"Table '{table_name}' not found in image at {self.path}.")
            return None

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            ad = importlib.import_module('anndata')
            anndata = ad.read_zarr(os.path.join(self.path, 'tables', table_name))
            # ... retain path information for combining anndata objects
            anndata.obs['image_path'] = self.path
            # ... create unique row identifier
            anndata.obs['unique_id'] = self.path + '/tables/' + table_name + '/' + anndata.obs.index.astype(str)

        if as_AnnData:
            return anndata
        else:
            df = pd.concat([anndata.obs['unique_id'],
                            anndata.obs['image_path'],
                            anndata.to_df()],
                            axis=1)
            return df

    # plotting methods -----------------------------------------------------------
    def plot(self,
             upper_left_yx: Optional[tuple[int]]=None,
             lower_right_yx: Optional[tuple[int]]=None,
             size_yx: Optional[tuple[int]]=None,
             coordinate_unit: str='micrometer',
             label_name: Optional[str]=None,
             label_value: Optional[Union[int, float, str]]=None,
             padding_pixels: int=0,
             pyramid_level: Optional[str]=None,
             pyramid_level_coord: Optional[str]=None,
             channels_labels: Optional[list[str]]=None,
             scalebar_micrometer: int=0,
             show_scalebar_label: bool=True,
             **kwargs: Any) -> None:
        """
        Plot an image.
         
        Plot a single image or a part of it, optionally selecting
        channels by label and extracting colors from the OMERO metadata
        using data at resolution `pyramid_level`.

        Parameters:
            upper_left_yx (tuple, optional): Tuple of (y, x) coordinates
                for the upper-left (lower) coordinates defining the region
                of interest.
            lower_right_yx (tuple, optional): Tuple of (y, x) coordinates
                for the lower-right (higher) coordinates defining the region
                of interest.
            size_yx (tuple, optional): Tuple of (size_y, size_x) defining
                the size of the region of interest.
            coordinate_unit (str): The unit of the image coordinates, for
                example 'micrometer' or 'pixel'.
            label_name (str, optional): The name of the a segmentation mask
                to be plotted semi-transparently over the intensity image.
                If `None`, just the intensity image is plotted.
            label_value (int, float, str, optional): The value of the label,
                for which bounding box coordinates should be extracted. 
                The highest resolution pyramid level will be used.If
                not `None`, `label_value` will automatically determine `upper_left_yx`,
                `lower_right_yx` and `size_yx`. Any values given to those or
                to `coordinate_unit` and `pyramid_level_coord` will be ignored.
            padding_pixels (int): The number of pixels to add to the final
                image region coordinates on both sides of each axis. Only used
                if `label_value` is not `None`.
            pyramid_level (str, optional): The pyramid level (resolution level),
                from which the intensity image should be extracted. If `None`,
                the lowest-resolution (highest) pyramid level will be selected.
            pyramid_level_coord (str, optional): An optional string giving the 
                image pyramid level to which the coordinates (`upper_left_yx`,
                `lower_right_yx` and `size_yx`) refer to if `coordinate_unit="pixel"`
                (it is ignored otherwise). By default, this is `None`, which
                indicates that coordinates refer to `pyramid_level`.
            channels_labels (list[str], optional): The labels of the image
                channel(s) to be plotted. This provides an alternative to
                selecting `channels` by index.
            scalebar_micrometer (int): If non-zero, add a scale bar corresponding
                to `scalebar_micrometer` to the image.
            show_scalebar_label (bool):  If `True`, add micrometer label to scale bar.
            **kwargs: Additional arguments for `plotting.plot_image`, for example
                'channels', 'channel_colors', 'channel_ranges', 'z_projection_method',
                etc. For a full list of available arguments, see
                [plotting.plot_image documentation](plotting.md#src.ez_zarr.plotting.plot_image).

        Examples:
            Plot the first channel in cyan from the whole intensity image `img`.

            >>> img.plot(channels = [0], channel_colors = ['cyan'])
        """
        # digest arguments
        assert label_name == None or label_name in self.label_names, (
            f"Unknown label_name ({label_name}), should be `None` or one of "
            ', '.join(self.label_names)
        )
        pyramid_level = self._digest_pyramid_level_argument(
            pyramid_level=pyramid_level,
            label_name=None
        )

        # import optional modules
        plotting = importlib.import_module('ez_zarr.plotting')

        # get channel indices
        if channels_labels != None:
            if 'channels' in kwargs:
                warnings.warn('`channels` will be ignored if `channels_labels` is given')
            kwargs['channels'] = self._digest_channels_labels(channels_labels)
        
        # extract `channel_colors`
        if 'channels' in kwargs and ('channel_colors' not in kwargs or  len(kwargs['channels']) != len(kwargs['channel_colors'])):
            print("extracting `channel_colors` from image metadata")
            kwargs['channel_colors'] = ['#' + self.channels[i]['color'] for i in kwargs['channels']]
        
        # extract `channel_ranges`
        if 'channels' in kwargs and ('channel_ranges' not in kwargs or len(kwargs['channels']) != len(kwargs['channel_ranges'])):
            print("setting `channel_ranges` based on length of `channels`")
            kwargs['channel_ranges'] = [[0.01, 0.95] for i in range(len(kwargs['channels']))]

        # get coordiantes by `label_value` if needed
        if label_value != None:
            if upper_left_yx != None or lower_right_yx != None or size_yx != None:
                warnings.warn("Ignoring provided coordinates since `label_value` was provided.")
            label_pyramid_level = self._find_path_of_highest_resolution_level(
                self.multiscales_labels[label_name]['datasets']
            )
            label_upper_left_yx, label_lower_right_yx = self._get_bounding_box_for_label_value(
                label_name=label_name,
                label_value=label_value,
                label_pyramid_level=label_pyramid_level,
                padding_pixels=0
            )
            if label_upper_left_yx is None:
                raise ValueError(f"No label value {label_value} found in label '{label_name}', pyramid level '{label_pyramid_level}'")
            upper_left_yx = convert_coordinates(
                coords_from=label_upper_left_yx[-2:],
                scale_from=self.get_scale(label_name=label_name,
                                          pyramid_level=label_pyramid_level)[-2:],
                scale_to=self.get_scale(pyramid_level=pyramid_level)[-2:]
            )
            lower_right_yx = convert_coordinates(
                coords_from=label_lower_right_yx[-2:],
                scale_from=self.get_scale(label_name=label_name,
                                          pyramid_level=label_pyramid_level)[-2:],
                scale_to=self.get_scale(pyramid_level=pyramid_level)[-2:]
            )
            size_yx = None
            coordinate_unit = 'pixel'
            pyramid_level_coord = None

            # padding
            upper_left_yx = tuple([max(0, upper_left_yx[i] - padding_pixels) for i in range(2)])
            lower_right_yx = tuple([min(self.array_dict[pyramid_level].shape[-2:][i], lower_right_yx[i] + padding_pixels) for i in range(2)])
        
        # get image and label arrays
        scale_img_spatial = self.get_scale(
            pyramid_level=pyramid_level,
            spatial_axes_only=True
        )
        if label_name == None:
            img = self.get_array_by_coordinate(
                label_name=None,
                upper_left_yx=upper_left_yx,
                lower_right_yx=lower_right_yx,
                size_yx=size_yx,
                coordinate_unit=coordinate_unit,
                pyramid_level=pyramid_level,
                pyramid_level_coord=pyramid_level_coord,
                as_NumPy=True
            )
            lab = None
        else:
            img, lab = self.get_array_pair_by_coordinate(
                label_name=label_name,
                upper_left_yx=upper_left_yx,
                lower_right_yx=lower_right_yx,
                size_yx=size_yx,
                coordinate_unit=coordinate_unit,
                pyramid_level=pyramid_level,
                pyramid_level_coord=pyramid_level_coord
            )

        # calculate scalebar length in pixel in x direction
        if scalebar_micrometer != 0:
            kwargs['scalebar_pixel'] = convert_coordinates(
                coords_from = (scalebar_micrometer,),
                scale_from = [1.0],
                scale_to = [self.get_scale(pyramid_level=pyramid_level, label_name=None)[-1]])[0]
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

# TODO:
# - plot_object(label_name, object_id)
