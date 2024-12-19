"""Wrap OME-Zarr filesets at different levels in classes.

Represent an OME-Zarr fileset as a class to give high-level
access to its contents.

Classes:
    Image: Contains a single `.zgroup`, typicallly a single image and possibly derived labels or tables.
"""

__all__ = ['Image', 'ImageList', 'create_name_row_col', 
           'create_name_plate_A01', 'import_plate']
__version__ = '0.3.5'
__author__ = 'Silvia Barbiero, Michael Stadler, Charlotte Soneson'


# imports -------------------------------------------------------------------
import os
import numpy as np
import zarr
import dask.array as da
import pandas as pd
import importlib
import warnings
from typing import Union, Optional, Callable, Any
from functools import cmp_to_key
from ez_zarr.utils import convert_coordinates, resize_image


# helper functions ------------------------------------------------------------
def create_name_row_col(ri: int, ci: int) -> str:
    """
    Create name by pasting row and column indices, separated by '_'.

    Parameters:
        ri (int): Row index (1-based)
        ci (int): Column index (1-based)
    
    Returns:
        str: Name constructed as f"{ri}_{ci}"

    Examples:
        >>> create_name_row_col(1, 2)
        '1_2'
    """
    return f"{ri}_{ci}"

def create_name_plate_A01(ri: int, ci: int) -> str:
    """
    Create name corresponding the wells in a microwell plate.

    Parameters:
        ri (int): Row index (1-based)
        ci (int): Column index (1-based)
    
    Returns:
        str: Name (`ci` always using two digits, with pre-fixed zeros)

    Examples:
        >>> create_name_plate_A01(3, 4)
        'C04'
    """
    return f"{chr(ord('A') + ri - 1)}{ci:02}"

def import_plate(path: str, image_name: str = '0') -> "ImageList":
    """
    Create an ImageList object from a OME-Zarr image set corresponding
    to a plate.

    Description:
        The image set is assumed to correspond to images corresponding to
        the wells of a microwell plate, and the folder names below `path`
        are expected to correspond to plate rows and columns, respectively
        (example the image in 'path/B/03/0' corresponds to well 'B03').
        Row and column names are read from the plate metadata
        (https://ngff.openmicroscopy.org/latest/index.html#plate-md).

    Parameters:
        path (str): Path to a folder containing the well images.
    
    Returns:
        ImageList object

    Examples:
        >>> imgL = import_plate('path/to/images')
        >>> imgL
        ImageList of 2 images
        paths: path/to/images/B/03/0, path/to/images/C/03/0
        names: B03, C03
    """

    # check arguments
    assert isinstance(path, str), f"path ({path}) must be a string."
    assert os.path.isdir(path), f"Path {path} does not exist."
    assert isinstance(image_name, str), f"image_name ({image_name}) must be a string."

    # list image paths from metadata
    img_paths = []
    row_index = []
    column_index = []
    well_names = []

    plate_zarr = zarr.open(store = path, mode='r')
    plate_metadata = plate_zarr.attrs.asdict()['plate']
    for well in plate_metadata['wells']:
        img_path = os.path.join(path, well['path'], image_name)
        if os.path.isdir(img_path):
            img_paths.append(img_path)
            row_name = plate_metadata['rows'][well['rowIndex']]['name']
            column_name = plate_metadata['columns'][well['columnIndex']]['name']
            well_names.append(row_name + column_name)
            row_index.append(ord(row_name) - ord('A') + 1)
            column_index.append(int(column_name))
        else:
            warnings.warn(f"Image path {img_path} does not exist.")
    if len(img_paths) == 0:
        raise Exception(f"No images found in {path}.")

    # build layout
    layout = pd.DataFrame({'row_index': row_index,
                           'column_index': column_index,
                           'img_paths': img_paths,
                           'well_names': well_names})
    layout = layout.sort_values(by=['row_index', 'column_index']).reset_index(drop=True)
    img_paths = layout.img_paths.tolist()
    well_names = layout.well_names.tolist()
    layout = layout.drop(['img_paths', 'well_names'], axis='columns')

    # set nrow, ncol
    known_plate_dims = [(2, 3), (4, 6), (8, 12), (16, 24)]
    mx_row = max(layout['row_index'])
    mx_column = max(layout['column_index'])
    nrow, ncol = known_plate_dims[min([i for i in range(len(known_plate_dims)) if known_plate_dims[i][0] >= mx_row and known_plate_dims[i][1] >= mx_column])]

    # create ImageList object
    imgL = ImageList(paths = img_paths, names = well_names,
                     layout=layout, nrow=nrow, ncol=ncol,
                     fallback_name_function=create_name_plate_A01)

    # return
    return imgL

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
        self.array_dict: dict[str, zarr.Array] = {x[0]: x[1] for x in self.zarr_group.arrays()}
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
        # ... load multiscales dictionaries
        self.multiscales_image: dict[str, Any] = self._load_multiscale_info(self.zarr_group, skip_checks)
        self.multiscales_labels: dict[str, dict[str, Any]] = {x: self._load_multiscale_info(self.zarr_group.labels[x], skip_checks) for x in self.label_names}
        # ... extract pyramid levels by decreasing resolution
        self.pyramid_levels_image: list[str] = Image._extract_paths_by_decreasing_resolution(self.multiscales_image['datasets'])
        self.pyramid_levels_labels: dict[str, list[str]] = {x: Image._extract_paths_by_decreasing_resolution(self.multiscales_labels[x]['datasets']) for x in self.label_names}
        # ... axes units
        self.axes_unit_image: str = self._load_axes_unit(self.multiscales_image)
        self.axes_unit_labels: dict[str, str] = {x: self._load_axes_unit(self.multiscales_labels[x]) for x in self.label_names}

        # load channel metadata
        # ... label dimensions, e.g. "czyx"
        self.channel_info_image: str = self._load_channel_info(self.multiscales_image)
        self.channel_info_labels: dict[str, str] = {x: self._load_channel_info(self.multiscales_labels[x]) for x in self.label_names}
        # ... store the number of image channels
        self.nchannels_image: int = self.array_dict[list(self.array_dict.keys())[0]].shape[self.channel_info_image.index('c')] if 'c' in self.channel_info_image else 0
        # ... store channel annotation from OMERO
        self.channels: list[dict[str, Any]] = []
        if 'omero' in self.zarr_group.attrs and 'channels' in self.zarr_group.attrs['omero']:
            self.channels = self.zarr_group.attrs['omero']['channels']
        elif self.nchannels_image > 0:
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
    def _extract_scale_spacings(dataset_dict: dict[str, Any]) -> list[Union[str, list[float]]]:
        if 'path' not in dataset_dict or 'coordinateTransformations' not in dataset_dict or 'scale' not in dataset_dict['coordinateTransformations'][0]:
            raise ValueError("could not extract zyx spacing from multiscale_info")
        return [dataset_dict['path'], dataset_dict['coordinateTransformations'][0]['scale']]
    
    @staticmethod
    def _compare_multiscale_dicts(a: dict[str, Any], b: dict[str, Any]) -> int:
        t1 = a['coordinateTransformations'][0]['scale']
        t2 = b['coordinateTransformations'][0]['scale']
        sum_of_votes = sum([1 if t1[i] > t2[i] else -1 if t1[i] < t2[i] else 0 for i in range(len(t1))])
        if sum_of_votes > 0:
            return 1  # t1 > t2 (t1 is lower resolution)
        elif sum_of_votes < 0:
            return -1 # t1 < t2 (t1 is higher resolution)
        else:
            return 0  # t1 = t2 (equal resolution)

    @staticmethod
    def _extract_paths_by_decreasing_resolution(datasets: list[dict[str, Any]]) -> list[str]:
        datasets_sorted = sorted(datasets, key=cmp_to_key(lambda x, y: Image._compare_multiscale_dicts(x, y)))
        return [str(x['path']) for x in datasets_sorted]

    def _find_path_of_lowest_resolution_level(self, label_name: Optional[str] = None) -> str:
        if label_name:
            if label_name in self.label_names:
                return self.pyramid_levels_labels[label_name][-1]
            else:
                raise ValueError(f"Label name '{label_name}' not found in Image object.")
        else:
            return self.pyramid_levels_image[-1]


    def _find_path_of_highest_resolution_level(self, label_name: Optional[str] = None) -> str:
        if label_name:
            if label_name in self.label_names:
                return self.pyramid_levels_labels[label_name][0]
            else:
                raise ValueError(f"Label name '{label_name}' not found in Image object.")
        else:
            return self.pyramid_levels_image[0]

    def _digest_pyramid_level_argument(self,
                                       pyramid_level=None,
                                       label_name=None,
                                       default_to='lowest') -> str:
        """
        [internal] Interpret a `pyramid_level` argument in the context of a given Image object.

        Parameters:
            pyramid_level (int, str or None): pyramid level, coerced to str. If None,
                the lowest-resolution pyramid level (or the highest-resolution one,
                if `default_to='highest'`) will be returned.
            label_name (str or None): defines what `pyramid_level` refers to. If None,
                it refers to the intensity image. Otherwise, it refers to a label with
                the name given by `label_name`. For example, to select the 'nuclei'
                labels, the argument would be set to `nuclei`.
            default_to (str): Defines what pyramid level to return if `pyramid_level`
                is `None`. Currently supported are `'lowest'` and `'highest'`.

        Returns:
            Integer index of the pyramid level.
        """
        if label_name is not None and label_name not in self.label_names:
            raise ValueError(f"invalid label name '{label_name}' - must be one of {self.label_names}")
        if pyramid_level == None: 
            # no pyramid level given -> pick according to `default_to`
            methods = {'lowest': self._find_path_of_lowest_resolution_level,
                       'highest': self._find_path_of_highest_resolution_level}
            pyramid_level = methods[default_to](label_name)
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
                             upper_left_yx: Optional[tuple[int, ...]]=None,
                             lower_right_yx: Optional[tuple[int, ...]]=None,
                             size_yx: Optional[tuple[int, ...]]=None,
                             coordinate_unit: str='micrometer',
                             label_name: Optional[str]=None,
                             pyramid_level: Optional[str]=None,
                             pyramid_level_coord: Optional[str]=None) -> list[tuple[int, ...]]:
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

            # convert and round to int
            upper_left_yx = tuple(int(round(x)) for x in convert_coordinates(upper_left_yx, scale_from[-2:], scale_to[-2:]))
            lower_right_yx = tuple(int(round(x)) for x in convert_coordinates(lower_right_yx, scale_from[-2:], scale_to[-2:]))

        # return
        return([upper_left_yx, lower_right_yx])
     
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
    
    def get_bounding_box_for_label_value(self,
                                         label_name: str,
                                         label_value: Union[int, float, str],
                                         label_pyramid_level: str,
                                         extend_pixels: Optional[int]=0,
                                         label_name_output: Optional[str]=None,
                                         pyramid_level_output: Optional[str]=None) -> Union[tuple[tuple[int, ...], tuple[int, ...]], tuple[None, None]]:
        """
        Given a label name and value, find the corner coordinates of the bounding box.

        Parameters:
            label_name (str): The name of the label image to extract the bounding
                box from.
            label_value (int, float, str): The value of the label to extract the
                bounding box for.
            label_pyramid_level (int, str): The pyramid level to extract the bounding
                box from.
            extend_pixels (int): The number of pixels to add to each side for each
                axis of the bounding box (before possible conversion to
                `label_name_output` and `pyramid_level_output`).
            label_name_output (str or None): The name of the label image to which
                the returned bounding box coordinates refer to. If `None` (the
                default), the coordinates refer to the intensity image.
            pyramid_level_output (str or None): The pyramid level to which the
                returned bounding box coordinates refer to. If `None` (the
                default), the coordinates refer to the highest resolution pyramid
                level.
        
        Returns:
            A tuple of upper-left and lower-right pixel coordinates for the bounding
            box containing the label value in the requested output space, or
            Throws an error if the label value is not found.
            For a 2D label image, this would be `((y1, x1), (y2, x2))`.
        
        Example:
            Get the bounding box for the 'nuclei' label value 1 from pyramid
            level 0,in the space of highest-resolution pyramid level of
            the intensity image:

            >>> img.get_bounding_box_for_label_value(label_name='nuclei', label_value=1, label_pyramid_level=0)
        """
        # digest arguments
        assert label_name in self.label_names, f'`label_name` must be in {self.label_names}'
        label_pyramid_level = self._digest_pyramid_level_argument(label_pyramid_level, label_name)
        assert isinstance(extend_pixels, int), '`extend_pixels` must be an integer scalar'
        assert isinstance(label_name_output, str) or label_name_output is None
        pyramid_level_output = self._digest_pyramid_level_argument(
            pyramid_level_output, label_name_output, default_to='highest')

        # get label array
        lab_arr = self.get_array_by_coordinate(label_name=label_name,
                                               pyramid_level=label_pyramid_level)

        # find bounding box
        value_coordinates = np.equal(lab_arr, label_value).nonzero()
        if len(value_coordinates[0]) == 0:
            raise ValueError(f'Label value {label_value} not found in label {label_name}')
        upper_left = tuple([min(x) for x in value_coordinates])
        lower_right = tuple([max(x) + 1 for x in value_coordinates])

        # padding
        upper_left = tuple([max(0, upper_left[i] - extend_pixels) for i in range(len(upper_left))])
        lower_right = tuple([min(lab_arr.shape[i], lower_right[i] + extend_pixels) for i in range(len(lower_right))])

        # convert to output space
        in_scale = self.get_scale(pyramid_level=label_pyramid_level,
                                  label_name=label_name,
                                  spatial_axes_only=True)
        out_scale = self.get_scale(pyramid_level=pyramid_level_output,
                                   label_name=label_name_output,
                                   spatial_axes_only=True)
        upper_left_out = convert_coordinates(coords_from=upper_left,
                                             scale_from=in_scale,
                                             scale_to=out_scale)
        lower_right_out = convert_coordinates(coords_from=lower_right,
                                              scale_from=in_scale,
                                              scale_to=out_scale)

        # return result
        return tuple([tuple([round(x) for x in upper_left_out]),
                      tuple([round(x) for x in lower_right_out])])
   
    def get_array_by_coordinate(self,
                                label_name: Optional[str]=None,
                                upper_left_yx: Optional[tuple[int]]=None,
                                lower_right_yx: Optional[tuple[int]]=None,
                                size_yx: Optional[tuple[int]]=None,
                                coordinate_unit: str='micrometer',
                                pyramid_level: Optional[str]=None,
                                pyramid_level_coord: Optional[str]=None,
                                as_NumPy: bool=False) -> Union[zarr.Array, np.ndarray]:
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
            The extracted array, either as an on-disk `zarr.Array`,
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
                                     label_name: Union[str, list[str]],
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
            label_name (str or list of str): The name(s) of the label image(s)
                to be extracted.
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
            A tuple of length two, with the first element corresponding to
            a `numpy.ndarray` with the extracted intensity array and the
            second element a dictionary with keys corresponding to the
            `label_name` and the values corresponding to label arrays.
            Label arrays are resized if necessary to match the intensity
            array.
        
        Examples:
            Obtain the whole image and matching 'organoids' label arrays:

            >>> img, lab = img.get_array_pair_by_coordinate(label_name = 'organoids')
        """
        # digest arguments
        if isinstance(label_name, str):
            label_name = [label_name]
        assert isinstance(label_name, list) and all([ln in self.label_names for ln in label_name]), (
            f"Unknown label_name(s) ({', '.join([ln for ln in label_name if ln not in self.label_names])}), should be one of "
            ', '.join(self.label_names)
        )
        pyramid_level = self._digest_pyramid_level_argument(
            pyramid_level=pyramid_level,
            label_name=None
        )
        if pyramid_level_coord is None:
            pyramid_level_coord = pyramid_level

        # get intensity image scale
        img_scale_spatial = self.get_scale(
            pyramid_level=pyramid_level,
            label_name=None,
            spatial_axes_only=True
        )

        # loop over label names
        lab_arr_dict = {}

        for lname in label_name:
            # find matching label pyramid level
            lab_scale_spatial_dict = {
                pl: self.get_scale(pyramid_level=pl, label_name=lname, spatial_axes_only=True) for pl in self.get_pyramid_levels(label_name=lname)
            }
            # ... filter out label scales with higher resolution than the intensity image
            lab_scale_spatial_dict = {pl: lab_scale_spatial for pl, lab_scale_spatial in lab_scale_spatial_dict.items() if all([lab_scale_spatial[i] >= img_scale_spatial[i] for i in range(len(lab_scale_spatial))])}
            if len(lab_scale_spatial_dict) == 0:
                raise ValueError(f"For the requested pyramid level ({pyramid_level}) of the intensity image, down-scaling of an available label ('{lname}') would be required. Down-scaling of labels is not supported - try selecting a higher-resolution intensity image.")

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
            img_arr = self.get_array_by_coordinate(
                upper_left_yx=imgpixel_upper_left_yx,
                lower_right_yx=imgpixel_lower_right_yx,
                size_yx=None,
                coordinate_unit='pixel',
                label_name=None,
                pyramid_level=pyramid_level, 
                as_NumPy=False
            )

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
                label_name=lname,
                pyramid_level=nearest_scale_pl
            ))

            # resize label if needed (correct non-matching scales or rounding errors)
            if lab_arr.shape[-2:] != img_arr.shape[-2:]:
                warnings.warn(f"For the requested pyramid level ({pyramid_level}) of the intensity image, no matching label ('{lname}') is available. Up-scaling the label using factor(s) {scalefact_yx}")
                lab_arr = resize_image(
                    im=lab_arr,
                    output_shape=img_arr.shape[(img_arr.ndim-lab_arr.ndim):],
                    im_type='label',
                    number_nonspatial_axes=sum([int(s not in ['z','y','x']) for s in self.channel_info_labels[lname]])
                )
            
            # store label array in dictionary
            lab_arr_dict[lname] = lab_arr

        # return arrays
        return tuple([np.array(img_arr), lab_arr_dict])


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

        if as_AnnData:
            return anndata
        else:
            return anndata.to_df()

    # zarr group methods ---------------------------------------------------------
    def tree(self, **kwargs: Any) -> Any:
        """Print zarr tree using zarr.hierarchy.Group.tree()."""
        return self.zarr_group.tree(**kwargs)

    # plotting methods -----------------------------------------------------------
    def plot(self,
             upper_left_yx: Optional[tuple[int]]=None,
             lower_right_yx: Optional[tuple[int]]=None,
             size_yx: Optional[tuple[int]]=None,
             coordinate_unit: str='micrometer',
             label_name: Optional[Union[list[str], str]]=None,
             label_value: Optional[Union[int, float, str]]=None,
             extend_pixels: int=0,
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
            label_name (str or list of str, optional): The segmentation
                mask name or names to be plotted semi-transparently over
                the intensity image. If this is a list of multiple names, the
                first one that is available for the image will be
                used. If `None`, just the intensity image is plotted.
            label_value (int, float, str, optional): The value of the label,
                for which bounding box coordinates should be extracted. 
                The highest resolution pyramid level will be used.If
                not `None`, `label_value` will automatically determine `upper_left_yx`,
                `lower_right_yx` and `size_yx`. Any values given to those or
                to `coordinate_unit` and `pyramid_level_coord` will be ignored.
            extend_pixels (int): The number of pixels to add to the final
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
                'show_label_values', 'label_text_colour', 'label_fontsize', etc.
                For a full list of available arguments, see
                [plotting.plot_image documentation](plotting.md#src.ez_zarr.plotting.plot_image).

        Examples:
            Plot the first channel in cyan from the whole intensity image `img`.

            >>> img.plot(channels = [0], channel_colors = ['cyan'])
        """
        # digest arguments
        if isinstance(label_name, list):
            label_name = [ln for ln in label_name if ln in self.label_names]
            if len(label_name) == 0:
                label_name = None
                print(f"None of given `label_name`s found in image {self.name}")
            else:
                label_name = label_name[0]
                print(f"Using label_name='{label_name}' for image {self.name}")
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

        # get coordinates by `label_value` if needed
        if label_value != None:
            if upper_left_yx != None or lower_right_yx != None or size_yx != None:
                warnings.warn("Ignoring provided coordinates since `label_value` was provided.")
            for label_pyramid_level in reversed(self.pyramid_levels_labels[label_name]):
                curr_array = self.get_array_by_coordinate(label_name=label_name, pyramid_level=label_pyramid_level)
                if da.isin(label_value, curr_array).compute():
                    break
            upper_left_yx, lower_right_yx = self.get_bounding_box_for_label_value(
                label_name=label_name,
                label_value=label_value,
                label_pyramid_level=label_pyramid_level,
                extend_pixels=0,
                label_name_output=None,
                pyramid_level_output=pyramid_level
            )
            upper_left_yx = upper_left_yx[-2:]
            lower_right_yx = lower_right_yx[-2:]
            size_yx = None
            coordinate_unit = 'pixel'
            pyramid_level_coord = None

            # padding
            upper_left_yx = tuple([max(0, upper_left_yx[i] - extend_pixels) for i in range(2)])
            lower_right_yx = tuple([min(self.array_dict[pyramid_level].shape[-2:][i], lower_right_yx[i] + extend_pixels) for i in range(2)])
        
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
            lab = lab[label_name] # extract label array from dict

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



# ImageList class -----------------------------------------------------------------
class ImageList:
    """Represents a list of OME-Zarr images."""

    # constructor and helper functions ----------------------------------------
    def __init__(self, paths: list[str],
                 names: Optional[list[str]]=None,
                 layout: Optional[Union[pd.DataFrame, str]]=None,
                 nrow: Optional[int]=None,
                 ncol: Optional[int]=None,
                 by_row: bool=False,
                 fallback_name_function: Callable[[int, int], str]=create_name_row_col) -> None:
        """
        Initializes an OME-Zarr image list from a list of paths, each
        containing a single zarr group, possibly with multiple resolution levels,
        derived labels or tables, but no further groups.

        If provided, the elements in `names` and/or the rows in `layout`
        need to be parallel to the element in `paths`.

        If `nrow` and/or `ncol` are provided without `layout`, the layout
        is automatically generated using `ImageList.set_layout()`.

        Parameters:
            paths (list of str): Paths containing the OME-Zarr images.
            names (list of str, optional): Optional names for the images. If
                `None`, the names are generated automatically using
                `fallback_name_function` (see below).
            layout (Pandas DataFrame or str, optional): Controls the plot layout.
                Either a data frame with at least the columns 'row_index' and
                'column_index'. The values in the 'row_index' and
                'column_index' are 1-based integers. Alternatively, a string
                specifying the type of layout to generate automatically.
                Currently supported layout types are 'grid'.
            nrow (int, optional): Number of rows in the layout.
            ncol (int, optional): Number of columns in the layout.
            by_row (bool, optional): If `True` and `layout` is not a data frame,
                the layout is generated by row instead of by column.
            fallback_name_function (Callable[[int, int], str]): Function that takes
                two integer parameters (the 1-based row and column index of an
                image) and returns a string representing the name for the image.
                Image names are for example used as titles in plots, and
                are only generated automatically as a fallback, if `names` is
                `None` or when plotting empty images for row/column indices
                that are not present in `layout`. `ez_zarr.ome_zarr` defines
                a few pre-defined functions that can be used, such as
                `create_name_row_col` and `create_name_plate_A01`.
                
        Returns:
            An `ImageList` object.
        
        Examples:
            Get an object corresponding to an image.

            >>> from ez_zarr import ome_zarr
            >>> listA = ome_zarr.ImageList(['path/to/image1', 'path/to/image2'])
        """

        self.paths: list = paths
        self.n_images = len(paths)
        self.layout: pd.DataFrame = pd.DataFrame({})
        self.nrow: int = 0
        self.ncol: int = 0
        self.fallback_name_function: Callable[[int, int], str] = fallback_name_function
        self.set_layout(layout=layout, nrow=nrow, ncol=ncol, by_row=by_row)
        if names is None:
            names = [self.fallback_name_function(ri, ci) for ri, ci in zip(self.layout['row_index'].to_list(), self.layout['column_index'].to_list())]
        self.names: list = names

        # check arguments
        assert len(self.names) == self.n_images, (
            f"Number of paths ({self.n_images}) and names ({len(self.names)}) must be the same."
        )
        if self.layout is not None:
            assert self.layout.shape[0] == self.n_images, (
                f"Number of paths ({self.n_images}) and layout rows ({self.layout.shape[0]}) must be the same."
            )
        
        # initialize image objects
        self.images = [Image(path=path, name=name) for path, name in zip(self.paths, self.names)]

    # length ------------------------------------------------------------------
    def __len__(self) -> int:
        """Returns the number of images."""
        return self.n_images
    
    # string representation ---------------------------------------------------
    def __str__(self) -> str:
        paths_compact = ", ".join(self.paths)
        if len(paths_compact) > 80:
            paths_compact = paths_compact[:80] + "..."
        names_compact = ", ".join(self.names)
        if len(names_compact) > 80:
            names_compact = names_compact[:80] + "..."
        return f"ImageList of {self.n_images} images\n  paths: {paths_compact}\n  names: {names_compact}\n"

    def __repr__(self) -> str:
        return str(self)

    # subsetting --------------------------------------------------------------
    def __getitem__(self, idx: Union[int, list[int], slice]) -> Union[Image, 'ImageList']:
        """Returns the image(s) at index `idx`."""

        if isinstance(idx, int):
            return self.images[idx]
        elif isinstance(idx, str):
            if len(set(self.names)) < self.n_images:
                raise ValueError(f"Image names are not unique - cannot subset by name.")
            return self.images[self.names.index(idx)]
        elif isinstance(idx, list) or isinstance(idx, slice):
            if all(isinstance(element, str) for element in idx):
                if len(set(self.names)) < self.n_images:
                    raise ValueError(f"Image names are not unique - cannot subset by names.")
                return self[[self.names.index(nm) for nm in idx]]
            else:
                if self.layout is not None:
                    layout_sub = self.layout.iloc[idx]
                else:
                    layout_sub = None
                return ImageList(paths=[self.paths[i] for i in idx],
                                 names=[self.names[i] for i in idx],
                                 layout=layout_sub)

    # call Image attributes ---------------------------------------------------
    def __getattr__(self, name):
        # remark: could catch "forbidden" attributes here
        try:
            attr = getattr(self.images[0], name)
        except AttributeError:
            raise AttributeError(f"'{type(self.images[0]).__name__}' objects have no attribute '{name}'")

        if callable(attr):
            def wrapper(*args, **kwargs):
                return [getattr(img, name)(*args, **kwargs) for img in self.images]
            return wrapper
        else:
            return [getattr(img, name) for img in self.images]


    # accessors ---------------------------------------------------------------
    def get_paths(self) -> list[str]:
        """Returns the paths of the images."""
        return self.paths
    
    def get_names(self) -> list[str]:
        """Returns the names of the images."""
        return self.names
    
    def get_layout(self) -> Optional[pd.DataFrame]:
        """Returns the layout of the images."""
        return self.layout
    
    # setting the layout ------------------------------------------------------
    @staticmethod
    def _create_layout_dataframe(nrow: int, ncol: int,
                       n: int, by_row: bool) -> pd.DataFrame:
        if by_row:
            ri = list(np.repeat(np.arange(nrow) + 1, ncol))[:n]
            ci = (list(np.arange(ncol) + 1) * nrow)[:n]
        else:
            ri = (list(np.arange(nrow) + 1) * ncol)[:n]
            ci = list(np.repeat(np.arange(ncol) + 1, nrow))[:n]
        layout = pd.DataFrame({'row_index': ri,
                               'column_index': ci})
        return layout

    def set_layout(self,
                   layout: Optional[Union[pd.DataFrame, str]]=None,
                   nrow: Optional[int]=None,
                   ncol: Optional[int]=None,
                   by_row: bool=False,
                   reset_names: bool=False) -> None:
        """Sets the `layout`, `nrow` and `ncol` properties of the `ImageList` object."""

        # set self.layout
        if isinstance(layout, pd.DataFrame):
            if layout.shape[0] != self.n_images:
                raise ValueError(f"Number of layout rows ({layout.shape[0]}) must be the same as the number of images ({self.n_images}).")
            self.layout = layout.copy()
        elif isinstance(layout, str):
            if nrow is not None or ncol is not None:
                warnings.warn("Ignoring `nrow` and `ncol` when layout is provided as a string.")
            if layout == 'grid':
                nrow = int(np.ceil(np.sqrt(self.n_images)))
                ncol = int(np.ceil(self.n_images / nrow))
                self.layout = self._create_layout_dataframe(nrow=nrow, ncol=ncol, n=self.n_images, by_row=by_row)
            else:
                raise ValueError(f"Layout provided as a string must be one of 'grid'.")
        elif layout is None:
            if ncol is None:
                if nrow is None:
                    nrow = int(np.ceil(np.sqrt(self.n_images)))
                ncol = int(np.ceil(self.n_images / nrow))
            if nrow is None:
                nrow = int(np.ceil(self.n_images / ncol))
            self.layout = self._create_layout_dataframe(nrow=nrow, ncol=ncol, n=self.n_images, by_row=by_row)
        
        # set self.nrow and self.ncol
        self.nrow = np.max(self.layout['row_index'].to_numpy()) if nrow is None else nrow
        self.ncol = np.max(self.layout['column_index'].to_numpy()) if ncol is None else ncol

        # set self.names
        if reset_names:
            self.names = [self.fallback_name_function(ri, ci) for ri, ci in zip(self.layout['row_index'].to_list(), self.layout['column_index'].to_list())]

    
    # plotting ----------------------------------------------------------------
    def plot(self,
             fig_title: Optional[str]=None,
             fig_width_inch: Optional[float]=None,
             fig_height_inch: Optional[float]=None,
             fig_dpi: int=200,
             fig_style: str='dark_background',
             **kwargs: Any) -> None:
        """
        Plot the images in the `ImageList`.
         
        The overall layout and the location of individual images are
        defined by `layout`, `nrow` and `ncol` properties. The plotting of
        individual images is can be controlled using `kwargs` and is
        performed using `Image.plot()`.

        Parameters:
            fig_title (str, optional): String scalar to use as overall figure title
                (default: do not add any title).
            fig_width_inch (float, optional): Figure width (in inches). If `None`,
                the number of rows multiplied by 2.0 will be used.
            fig_height_inch (float, optional): Figure height (in inches). If `None`,
                the number of columns multiplied by 2.0 will be used.
            fig_dpi (int): Figure resolution (dots per inch).
            fig_style (str): Style for the figure. Supported are 'brightfield', which
                is a special mode for single-channel brightfield microscopic images
                (it will automatically set `channels=[0]`, `channel_colors=['white']`
                `z_projection_method='minimum'` and `fig_style='default'`), and any
                other styles that can be passed to `matplotlib.pyplot.style.context`
                (default: 'dark_background')
            **kwargs: Additional arguments for `Image.plot_image`, for example
                'label_name', 'pyramid_level', 'scalebar_micrometer', 'show_scalebar_label',
                'channels', 'channel_colors', 'channel_ranges', 'z_projection_method',
                'show_label_values', 'label_text_colour', 'label_fontsize', etc.
                For a full list of available arguments, see
                [ome_zarr.Image.plot documentation](ome_zarr.md#src.ez_zarr.ome_zarr.Image.plot)
                and the underlying
                [plotting.plot_image documentation](plotting.md#src.ez_zarr.plotting.plot_image).

        Examples:
            Plot the intensity images in the `ImageList` `imgL`.

            >>> imgL.plot()
        """

        # digest arguments
        fig_width_inch = fig_width_inch if fig_width_inch is not None else self.ncol * 2.0
        fig_height_inch = fig_height_inch if fig_height_inch is not None else self.nrow * 2.0

        # import optional modules
        plt = importlib.import_module('matplotlib.pyplot')

        # get the maximal image y,x coordinates
        channel_info = [img.channel_info_image for img in self.images]
        assert all([channel_info[0] == ci for ci in channel_info]), f"Not all images have the same channel info: {', '.join(channel_info)}"
        spatial_dims = [i for i in range(len(channel_info[0])) if channel_info[0][i] in ['y', 'x']]
        img_dims = [img.get_array_by_coordinate().shape for img in self.images]
        img_dims_spatial = [[d[i] for i in spatial_dims] for d in img_dims]
        max_yx = np.max(np.stack(img_dims_spatial), axis=0)
        kwargs['pad_to_yx'] = max_yx

        # adjust parameters for brightfield images
        if fig_style == 'brightfield':
            kwargs['channels'] = [0]
            kwargs['channel_colors'] = ['white']
            kwargs['z_projection_method'] = 'minimum'
            kwargs['pad_value'] = 1
            fig_style = 'default'
            empty_well = np.ones(max_yx)
        else:
            kwargs['pad_value'] = 0
            empty_well = np.zeros(max_yx)

        # create figure
        with plt.style.context(fig_style):
            fig = plt.figure(figsize=(fig_width_inch, fig_height_inch))
            fig.set_dpi(fig_dpi)
            if fig_title is not None:
                fig.suptitle(fig_title, size='xx-large') # default title size: 'large'
            
            # loop over images
            kwargs['call_show'] = False # don't create new figures for each image
            for ri in range(self.nrow):
                for ci in range(self.ncol):
                    plt.subplot(self.nrow, self.ncol, ri * self.ncol + ci + 1)

                    # check if we have an image for this row and column
                    condition = (self.layout['row_index'] == ri + 1) & (self.layout['column_index'] == ci + 1)
                    n_matching = np.sum(condition.to_numpy())

                    if n_matching > 1:
                        # more than one image
                        raise Exception(f"More than one image at row {ri + 1}, column {ci + 1}")
                    
                    elif n_matching == 1:
                        # exactly one image, plot it
                        i = self.layout[condition].index.to_list()[0]
                        kwargs['title'] = self.names[i]
                        self.images[i].plot(**kwargs)
                    
                    else:
                        # plot empty well
                        plt.imshow(empty_well, cmap='gray', vmin=0, vmax=1)
                        if 'axis_style' in kwargs and kwargs['axis_style'] != 'none':
                            plt.xticks([]) # remove axis ticks
                            plt.yticks([])
                        else:
                            plt.axis('off')
                        plt.title(self.fallback_name_function(ri + 1, ci + 1))
            fig.tight_layout()
            plt.show()
            plt.close()


