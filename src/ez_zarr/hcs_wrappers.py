"""Wrap OME-Zarr filesets in a class.

Represent an OME-Zarr fileset as a class to give high-level
access to its contents.

Classes:
    FractalZarr: Contains a single `.zarr` fileset, typically a plate.
    FractalZarrSet: Contains one or several `.zarr` filesets, typically a plate
        (4 dimensional data) and a maximum-intensity projection derived from it.
"""

__all__ = ['FractalZarr', 'FractalZarrSet']
__version__ = '0.1.5'
__author__ = 'Silvia Barbiero, Michael Stadler'


# imports ---------------------------------------------------------------------
import os
import numpy as np
import zarr
import dask.array
import pandas as pd
import importlib
import warnings
import random
from typing import Union, Optional, Any


# FractalZarr class ---------------------------------------------------------------
class FractalZarr:
    """Represents an OME-Zarr fileset."""

    # constructor and helper functions ----------------------------------------
    def __init__(self, zarr_path: str, name: Optional[str]=None) -> None:
        """
        Initializes an OME-Zarr fileset (.zarr) from its path.
        Typically, the fileset represents a single assay plate, and 
        we assume that the structures (pyramid levels, labels, tables, etc.)
        are consistent across wells.

        Parameters:
            zarr_path (str): Path containing the plate ome-zarr fileset.
            name (str, optional): Optional name for the plate.
        
        Examples:
            Get an object corresponding to a plate.

            >>> from ez_zarr import hcs_wrappers
            >>> plateA = hcs_wrappers.FractalZarr('path/to/plate.zarr')
            >>> plateA

            This will print information about the plate.
        """

        self.path: str = zarr_path
        self.name: str = ''
        if name:
            self.name = name
        else:
            self.name = os.path.basename(self.path)
        self.__top: zarr.Group = zarr.open_group(store=self.path, mode='r')
        if not 'plate' in self.__top.attrs:
            raise ValueError(f"{self.name} does not contain a zarr fileset with a 'plate'")
        self.acquisitions: list = self.__top.attrs['plate']['acquisitions']
        self.columns: list[dict] = self.__top.attrs['plate']['columns']
        self.rows: list[dict] = self.__top.attrs['plate']['rows']
        self.wells: list[dict] = self.__top.attrs['plate']['wells']
        # images
        self.image_names: list[str] = self._load_image_names()
        self.channels: list[dict] = self._load_channel_info()
        self.multiscales_images: dict[str, Any] = self._load_multiscale_info('images')
        self.level_paths_images: dict[str, list[str]] = {im: [x['path'] for x in self.multiscales_images[im]['datasets']] for im in self.image_names}
        self.level_zyx_spacing_images: dict[str, list[list[float]]] = {im: [x['coordinateTransformations'][0]['scale'][1:] for x in self.multiscales_images[im]['datasets']] for im in self.image_names} # convention: unit is micrometer
        self.level_zyx_scalefactor: dict[str, np.ndarray] = {im: np.divide(self.level_zyx_spacing_images[im][1], self.level_zyx_spacing_images[im][0]) for im in self.image_names}
        # labels
        self.label_names: list[str] = self._load_label_names()
        self.multiscales_labels: dict[str, Any] = self._load_multiscale_info('labels')
        self.level_paths_labels: dict[str, list[str]] = {lab: [x['path'] for x in self.multiscales_labels[lab]['datasets']] for lab in self.label_names}
        self.level_zyx_spacing_labels: dict[str, list[list[float]]] = {lab: [x['coordinateTransformations'][0]['scale'] for x in self.multiscales_labels[lab]['datasets']] for lab in self.label_names} # convention: unit is micrometer
        # tables
        self.table_names: list[str] = self._load_table_names()

    def _load_channel_info(self) -> list:
        """[internal] Load info about available channels."""
        well_path = self.wells[0]['path']
        well_group = self.__top[os.path.join(well_path, self.image_names[0])]
        if not 'omero' in well_group.attrs or not 'channels' in well_group.attrs['omero']:
            raise ValueError(f"no channel info found in well {well_path}, image {self.image_names[0]}")
        return well_group.attrs['omero']['channels']
    
    def _load_multiscale_info(self, target: str) -> dict[str, Any]:
        """[internal] Load info about available scales in target (images or labels)."""
        well_path = self.wells[0]['path']
        if target == 'images':
            pre = ''
            els = self.image_names
        else:
            pre = os.path.join(self.image_names[0], 'labels')
            els = self.label_names
        info = {}
        for el in els:
            el_group = self.__top[os.path.join(well_path, pre, el)]
            if not 'multiscales' in el_group.attrs:
                raise ValueError(f"no multiscale info found in {target} {el}")
            info[el] = el_group.attrs['multiscales'][0]
        return info
       
    def _load_image_names(self) -> list[str]:
        """[internal] Load image names (available images)."""
        well_path = self.wells[0]['path']
        ims = [im for im in os.listdir(os.path.join(self.path, well_path)) if not im.startswith('.')]
        return ims

    def _load_label_names(self) -> list[str]:
        """[internal] Load label names (available segmentations)."""
        well = self.wells[0]['path']
        label_path = os.path.join(well, '0', 'labels')
        if label_path in self.__top:
            return self.__top[label_path].attrs['labels']
        else:
            return []

    def _load_table_names(self) -> list[str]:
        """[internal] Load table names (can be extracted using .get_table())."""
        well = self.wells[0]['path']
        table_path = os.path.join(well, '0', 'tables')
        if table_path in self.__top:
            return self.__top[table_path].attrs['tables']
        else:
            return []
    
    # utility functions -------------------------------------------------------
    def _digest_well_argument(self, well=None, as_path=True):
        """[internal] Interpret a single `well` argument in the context of a given FractalZarr object."""
        if not well:
            # no well given -> pick first one
            return self.wells[0]['path']
        elif as_path:
            return os.path.join(well[:1].upper(), well[1:])
        else:
            return well

    def _digest_include_wells_argument(self, include_wells: Union[str, list[str]]=[]) -> list[str]:
        """[internal] Interpret an `include_wells` argument in the context of a given FractalZarr object."""
        if isinstance(include_wells, str):
            include_wells = [include_wells]
        if len(include_wells) == 0: 
            # no wells given -> include all wells
            include_wells = [x['path'] for x in self.wells]
        else:
            # transform well names from 'B03' format to path format 'B/03'
            include_wells = [self._digest_well_argument(w) for w in include_wells]
        return include_wells

    def _digest_pyramid_level_argument(self, pyramid_level=None, pyramid_ref=None) -> int:
        """
        [internal] Interpret a `pyramid_level` argument in the context of a given FractalZarr object.

        Parameters:
            pyramid_level (int, str or None): pyramid level, coerced to str. If None, the
                last pyramid level (typically the lowest-resolution one) will be returned.
            pyramid_ref (tuple(str, str)): defines what `pyramid_level` refers to. If None,
                the first image is used: `('image', '0')` (assuming that the name of the
                first image is '0'). To select the 'nuclei' labels, the arugment would be
                set to `('label', 'nuclei')`.

        Returns:
            Integer index of the pyramid level.
        """
        if pyramid_ref == None:
            # no pyramid reference given -> pick the first image
            pyramid_ref = ('image', self.image_names[0])
        if pyramid_level == None: 
            # no pyramid level given -> pick lowest resolution one
            if pyramid_ref[0] == 'image':
                pyramid_level = int(self.level_paths_images[pyramid_ref[1]][-1])
            else:
                pyramid_level = int(self.level_paths_labels[pyramid_ref[1]][-1])
        else:
            # make sure it is an integer
            pyramid_level = int(pyramid_level)
        return pyramid_level
    
    def _calculate_regular_grid_coordinates(self, y, x, num_y=10, num_x=10):
        """
        [internal] Calculate the cell coordinates for a regular rectangular grid of total size (y, x)
        by splitting the dimensions into num_y and num_x cells.
        
        Returns a list of (y_start, y_end, x_start, x_end) tuples. The coordinates are
        inclusive at the start and exclusive at the end.

        All returned grid cells are guaranteed to be of equal size, but a few pixels in the
        last row or column may not be included if y or x is not divisible by `num_y` or `num_x`.
        """
        y_step = y // num_y
        x_step = x // num_x

        grid_coords = [] # list of (y_start, y_end, x_start, x_end)
        for i in range(num_y):
            for j in range(num_x):
                y_start = i * y_step
                # y_end = (i + 1) * y_step if i != num_y - 1 else y
                y_end = (i + 1) * y_step # possibly miss some at the highest i
                x_start = j * x_step
                # x_end = (j + 1) * x_step if j != num_x - 1 else x
                x_end = (j + 1) * x_step # possibly miss some at the highest i

                grid_coords.append((y_start, y_end, x_start, x_end))
        
        return grid_coords

    # string representation ---------------------------------------------------
    def __str__(self):
        nwells = len(self.wells)
        nch = len(self.channels)
        chlabs = ', '.join([x['label'] for x in self.channels])
        npl = len(self.multiscales_images[self.image_names[0]]['datasets'])
        segnames = ', '.join(self.label_names)
        tabnames = ', '.join(self.table_names)
        return f"FractalZarr {self.name}\n  path: {self.path}\n  n_wells: {nwells}\n  n_channels: {nch} ({chlabs})\n  n_pyramid_levels: {npl}\n  pyramid_zyx_scalefactor: {self.level_zyx_scalefactor}\n  full_resolution_zyx_spacing (µm): {self.level_zyx_spacing_images[self.image_names[0]][0]}\n  segmentations: {segnames}\n  tables (measurements): {tabnames}\n"
    
    def __repr__(self):
        return str(self)
    
    # accessors ---------------------------------------------------------------
    def get_path(self) -> str:
        """Get the path of an OME-Zarr fileset.
        
        Returns:
            The path to the OME-Zarr fileset.
        """
        return self.path

    def get_wells(self, simplify: bool=False) -> Union[list[dict[str, Any]], list[str]]:
        """Get info on wells in an OME-Zarr fileset.

        Parameters:
            simplify (bool): If `True`, the well names are returned in human readable form (e.g. 'B03').
        
        Returns:
            A list of wells in the plate, either name strings (if `simplify=True`) or dicts with well attributes.
        """
        if simplify:
            return [w['path'].replace('/', '') for w in self.wells]
        else:
            return self.wells

    def get_channels(self) -> list:
        """Get info on channels in an OME-Zarr fileset.
        
        Returns:
            A list of dicts with information on channels.
        """
        return self.channels
    
    def get_label_names(self) -> list:
        """Get list of label names in an OME-Zarr fileset.
        
        Returns:
            A list of label names (str) available in the plate.
        """
        return self.label_names

    def get_table_names(self) -> list:
        """Get list of table names in an OME-Zarr fileset.
        
        Returns:
            A list of table names (str) available in the plate.
        """
        return self.table_names

    # query methods -----------------------------------------------------------
    def get_table(self,
                  table_name: str,
                  include_wells: Union[str, list[str]]=[],
                  as_AnnData: bool=False) -> Any:
        """Extract table for wells in an OME-Zarr fileset.
        
        Parameters:
            table_name (str): The name of the table to extract.
            include_wells (str or list): List of well names to include. If empty `[]`, all wells are included.
            as_AnnData (bool): If `True`, the table is returned as an `AnnData` object, otherwise it is converted to a `pandas.DataFrame`.
        
        Returns:
            The extracted table, either as an `anndata.AnnData` object if `as_AnnData=True`, and as a `pandas.DataFrame` otherwise.
        
        Examples:
            Get a table with coordinates of fields of view:

            >>> plateA.get_table(table_name='FOV_ROI_table')
        """
        include_wells = self._digest_include_wells_argument(include_wells)

        table_paths = [os.path.join(w, '0', 'tables', table_name) for w in include_wells]
        # remark: warn if not all well have the table?
        table_paths = [p for p in table_paths if p in self.__top]

        if len(table_paths) == 0:
            return None

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            ad = importlib.import_module('anndata')
            anndata_list = [ad.read_zarr(os.path.join(self.path, p)) for p in table_paths]
            # ... retain well information when combining anndata objects
            for ann, w in zip(anndata_list, include_wells):
                ann.obs['well'] = w.replace('/', '')
            anndata_combined = ad.concat(anndata_list, index_unique=None, keys=[w.replace('/', '') for w in include_wells])

        if as_AnnData:
            return anndata_combined
        else:
            df_combined = pd.concat([anndata_combined.obs['well'],
                                     anndata_combined.to_df()],
                                     axis=1)
            return df_combined

    def get_image_ROI(self,
                      upper_left_yx: Optional[tuple[int]]=None,
                      lower_right_yx: Optional[tuple[int]]=None,
                      size_yx: Optional[tuple[int]]=None,
                      well: Optional[str]=None,
                      image_name: str='0',
                      pyramid_level: Optional[int]=None,
                      pyramid_level_coord: Optional[int]=None,
                      as_NumPy: bool=False) -> Union[dask.array.Array, np.ndarray]:
        """
        Extract a region of interest from a well image by coordinates.

        None or exactly two of `upper_left_yx`, `lower_right_yx` and `size_yx` need to be given.
        If none are given, it will return the full image (the whole well).
        Otherwise, `upper_left_yx` contains the lower indices than `lower_right_yx`
        (origin on the top-left, zero-based coordinates), and each of them is
        a tuple of (y, x). No z coordinate needs to be given, all z planes are returned
        if there are several ones.

        Parameters:
            upper_left_yx (tuple): Tuple of (y, x) coordinates for the upper-left (lower) coordinates
                defining the region of interest.
            lower_right_yx (tuple): Tuple of (y, x) coordinates for the lower-right (higher) coordinates defining the region of interest.
            size_yx (tuple): Tuple of (size_y, size_x) defining the size of the region of interest.
            well (str): The well (e.g. 'B03') from which an image should be extracted.
            image_name (str): The name of the image in `well` to extract from. Default: '0'.
            pyramid_level (int): The pyramid level (resolution level), from which the image
                should be extracted. If `None`, the lowest-resolution (highest) pyramid level
                will be selected.
            pyramid_level_coord (int): An optional integer scalar giving the image pyramid level
                to which the coordinates (`upper_left_yx`, `lower_right_yx` and `size_yx`)
                refer to. By default, this is `None`, which will use `pyramid_level`.
            as_NumPy (bool): If `True`, return the image as 4D `numpy.ndarray` object (c,z,y,x).
                Otherwise, return the (on-disk) `dask` array of the same dimensions.
        
        Returns:
            The extracted image, either as a `dask.array.Array` on-disk array, or as an in-memory `numpy.ndarray` if `as_NumPy=True`.
        
        Examples:
            Obtain the image of the lowest-resolution for the full well 'A02':

            >>> plateA.get_image_ROI(well='A02')
        """
        # digest arguments
        well = self._digest_well_argument(well)
        pyramid_level = self._digest_pyramid_level_argument(pyramid_level, ('image', image_name))

        # load image
        img_path = os.path.join(self.path, well, image_name, str(pyramid_level))
        img = dask.array.from_zarr(img_path)

        # calculate corner coordinates and subset if needed
        # (images are always of 4D shape c,z,y,x)
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
            if pyramid_level_coord != None and pyramid_level != pyramid_level_coord:
                upper_left_yx = self.convert_pixel_to_pixel(zyx=((0,) + upper_left_yx),
                                                            pyramid_level_from=pyramid_level_coord,
                                                            pyramid_level_to=pyramid_level,
                                                            pyramid_ref_from=('image', image_name),
                                                            pyramid_ref_to=('image', image_name))[1:]
                lower_right_yx = self.convert_pixel_to_pixel(zyx=((0,) + lower_right_yx),
                                                             pyramid_level_from=pyramid_level_coord,
                                                             pyramid_level_to=pyramid_level,
                                                             pyramid_ref_from=('image', image_name),
                                                             pyramid_ref_to=('image', image_name))[1:]
            img = img[:,
                      :,
                      slice(upper_left_yx[0], lower_right_yx[0] + 1),
                      slice(upper_left_yx[1], lower_right_yx[1] + 1)]
        elif num_unknowns != 3:
            raise ValueError("Either none or two of `upper_left_yx`, `lower_right_yx` and `size_yx` have to be given")

        # convert if needed and return
        if as_NumPy:
            img = np.array(img)
        return img

    def get_image_table_idx(self,
                            table_name: str,
                            table_idx: int,
                            well: Optional[str]=None,
                            pyramid_level: Optional[int]=None,
                            as_NumPy: bool=False) -> Union[dask.array.Array, np.ndarray]:
        """
        Extract a region of interest from a well image by table name and row index.

        Bounding box coordinates will be automatically obtained from the table
        `table_name` and row `row_idx` (zero-based row index).
        All z planes are returned if there are several ones.

        Parameters:
            table_name (str): The name of the table containing object coordinates in columns
                `x_micrometer`, `y_micrometer`, `len_x_micrometer` and `len_y_micrometer`.
            table_idx (int): The zero-based row index for the object to be extracted.
            well (str): The well (e.g. 'B03') from which an image should be extracted.
            pyramid_level (int): The pyramid level (resolution level), from which the image
                should be extracted. If `None`, the lowest-resolution (highest) pyramid level
                will be selected.
            as_NumPy (bool): If `True`, return the image as 4D `numpy.ndarray` object (c,z,y,x).
                Otherwise, return the (on-disk) `dask` array of the same dimensions.
        
        Returns:
            The extracted image, either as a `dask.array.Array` on-disk array, or as an in-memory `numpy.ndarray` if `as_NumPy=True`.
        
        Examples:
            Obtain the image of the first object in table `nuclei_ROI_table` in well 'A02':

            >>> plateA.get_image_table_idx(table_name='nuclei_ROI_table', table_idx=0, well='A02')
        """
        # digest arguments
        assert table_name in self.table_names, f"Unknown table {table_name}, should be one of " + ', '.join(self.table_names)
        well = self._digest_well_argument(well, as_path=False)
        pyramid_level = self._digest_pyramid_level_argument(pyramid_level)

        # extract table
        df = self.get_table(table_name=table_name, include_wells=well, as_AnnData=False)
        required_columns = ['x_micrometer', 'y_micrometer', 'len_x_micrometer', 'len_y_micrometer']
        assert all(column in set(df.columns) for column in required_columns), f"Missing columns: {set(required_columns) - set(df.columns)}"
        assert table_idx < len(df), f"table_idx ({table_idx}) needs to be less than " + str(len(df))

        # get bounding box coordinates
        ul = self.convert_micrometer_to_pixel((0,
                                               df['y_micrometer'].iloc[table_idx],
                                               df['x_micrometer'].iloc[table_idx]),
                                               pyramid_level=pyramid_level,
                                               pyramid_ref=('image', '0'))
        hw = self.convert_micrometer_to_pixel((0,
                                               df['len_y_micrometer'].iloc[table_idx],
                                               df['len_x_micrometer'].iloc[table_idx]),
                                               pyramid_level=pyramid_level,
                                               pyramid_ref=('image', '0'))


        # load image
        img = self.get_image_ROI(upper_left_yx=ul[1:],
                                 size_yx=hw[1:],
                                 well=well,
                                 pyramid_level=pyramid_level,
                                 as_NumPy=as_NumPy)

        return img

    def get_image_grid_ROIs(self,
                            well: Optional[str]=None,
                            pyramid_level: Optional[int]=None,
                            lowres_level: Optional[int]=None,
                            num_y: int=10,
                            num_x: int=10,
                            num_select: int=9,
                            sample_method: str='sum',
                            channel: int=0,
                            seed: int=42,
                            as_NumPy: bool=False) -> Union[tuple[list[tuple[int]], list[dask.array.Array]], tuple[list[tuple[int]], list[np.ndarray]]]:
        """
        Split a well image into a regular grid and extract a subset of grid cells (all z planes if several).

        `num_y` and `num_x` define the grid by specifying the number of cells in y and x.
        `num_select` grid cells are picked from the total number and returned as a list.
        All returned grid cells are guaranteed to be of equal size, but a few pixels from the image
        may not be included in grid cells of the last row or column if the image shape
        is not divisible by `num_y` or `num_x`.

        Parameters:
            well (str):  The well (e.g. 'B03') from which an image should be extracted.
            pyramid_level (int): The pyramid level (resolution level), from which the
                selected image grid cells should be returned. If `None`, the lowest-resolution
                (highest) pyramid level will be selected.
            lowres_level (int): Similar to `pyramid_level`, but defining the resolution
                for calculating the `sample_method`. Calculations on low-resolution
                images are faster and often result in identical ordering of grid cells,
                so that high-resolution images can be returned by `pyramid_level` without making
                their sampling slower.
            num_y (int): The size of the grid in y.
            num_x (int): The size of the grid in x.
            num_select (int): The number of grid cells to return as images.
            sample_method (str): Defines how the `num_select` cells are selected. Possible values are:
                - 'sum': order grid cells decreasingly by the sum of `channel` (working on `lowres_level`)
                - 'var': order grid cells decreasingly by the variance of `channel` (working on `lowres_level`)
                - 'random': order grid cells randomly (use `seed`)
            channel (int): Selects the channel on which `sample_method` is calculated.
            seed (int): Used in `random.seed()` to make sampling for `sample_method='random'` reproducible.
            as_NumPy (bool): If `True`, return the grid cell image as `numpy.ndarray` objects with
                shapes (c,z,y,x). Otherwise, return the (on-disk) `dask` arrays of the same dimensions.
        
        Returns:
            A tuple of two lists (coord_list, img_list), each with `num_select` elements 
                corresponding to the coordinates and images of selected grid cells. 
                The coordinates are tuples of the form  (y_start, y_end, x_start, x_end).

        Examples:
            Obtain grid cells with highest signal sum in channel 0 from well 'A02':

            >>> plateA.get_image_grid_ROIs(well='A02')
        """
        # digest arguments
        well = self._digest_well_argument(well)
        pyramid_level = self._digest_pyramid_level_argument(pyramid_level)
        lowres_level = self._digest_pyramid_level_argument(lowres_level)
        assert num_select <= num_x * num_y, f"num_select ({num_select}) needs to be less or equal to num_x * num_y" + str(num_x * num_y)

        # load image (convention: single field of view per well -> '0')
        img_path = os.path.join(self.path, well, '0', str(pyramid_level))
        img = dask.array.from_zarr(img_path)
        img_path_lowres = os.path.join(self.path, well, '0', str(lowres_level))
        img_lowres = dask.array.from_zarr(img_path_lowres)

        # calculate grid coordinates as a list of (y_start, y_end, x_start, x_end)
        # (images are always of 4D shape c,z,y,x)
        ch, z, y, x = img.shape
        grid = self._calculate_regular_grid_coordinates(y=y, x=x,
                                                        num_y=num_y,
                                                        num_x=num_x)
        ch_lr, z_lr, y_lr, x_lr = img_lowres.shape
        grid_lowres = self._calculate_regular_grid_coordinates(y=y_lr, x=x_lr,
                                                               num_y=num_y,
                                                               num_x=num_x)

        # select and extract grid cells
        sel_coords = []
        sel_img_cells = []

        if sample_method == 'sum':
            grid_values = [
                dask.array.sum(
                    img_lowres[slice(channel, channel + 1),
                               :,
                               slice(grid_lowres[i][0], grid_lowres[i][1]),
                               slice(grid_lowres[i][2], grid_lowres[i][3])]
                ) for i in range(len(grid_lowres))
            ]
            idx_sorted = list(np.argsort(np.array(grid_values)))
            sel_coords = [grid[i] for i in idx_sorted[-num_select:]]
        elif sample_method == 'var':
            grid_values = [
                dask.array.var(
                    img_lowres[slice(channel, channel + 1),
                               :,
                               slice(grid_lowres[i][0], grid_lowres[i][1]),
                               slice(grid_lowres[i][2], grid_lowres[i][3])]
                ) for i in range(len(grid_lowres))
            ]
            idx_sorted = list(np.argsort(np.array(grid_values)))
            sel_coords = [grid[i] for i in idx_sorted[-num_select:]]
        elif sample_method == 'random':
            random.seed(seed)
            sel_coords = random.sample(grid, num_select)
        else:
            raise ValueError("'sample_method' must be one of 'sum', 'var' or 'random'")
        for i in range(num_select):
            img_cell = img[:,
                           :,
                           slice(sel_coords[i][0], sel_coords[i][1]),
                           slice(sel_coords[i][2], sel_coords[i][3])]
            if as_NumPy:
                img_cell = np.array(img_cell)
            sel_img_cells.append(img_cell)

        return (sel_coords, sel_img_cells)

    def get_label_ROI(self,
                      label_name: str,
                      well: Optional[str]=None,
                      image_name: str='0',
                      pyramid_level: Optional[int]=None,
                      pyramid_level_coord: Optional[int]=None,
                      upper_left_yx: Optional[tuple[int]]=None,
                      lower_right_yx: Optional[tuple[int]]=None,
                      size_yx: Optional[tuple[int]]=None,
                      as_NumPy: bool=False) -> Union[dask.array.Array, np.ndarray]:
        """
        Extract a region of interest from a label mask (segmentation) by coordinates.

        None or exactly two of `upper_left_yx`, `lower_right_yx` and `size_yx` need to be given.
        If none are given, it will return the full image (the whole well).
        Otherwise, `upper_left_yx` contains the lower indices than `lower_right_yx`
        (origin on the top-left, zero-based coordinates), and each of them is
        a tuple of (y, x). No z coordinate needs to be given, all z planes are returned
        if there are several ones.

        Parameters:
            label_name (str): The name of the segmentation
            well (str): The well (e.g. 'B03') from which an image should be extracted.
            image_name (str): The name of the image in `well` to extract labels from. Default: '0'.
            pyramid_level (int): The pyramid level (resolution level), from which the image
                should be extracted. If `None`, the lowest-resolution (highest) pyramid level
                will be selected.
            pyramid_level_coord (int): An optional integer scalar giving the image pyramid level
                to which the coordinates (`upper_left_yx`, `lower_right_yx` and `size_yx`)
                refer to. By default, this is `None`, which will use `pyramid_level`.
            upper_left_yx (tuple): Tuple of (y, x) coordinates for the upper-left (lower) coordinates
                defining the region of interest.
            lower_right_yx (tuple): Tuple of (y, x) coordinates for the lower-right (higher) coordinates defining the region of interest.
            size_yx (tuple): Tuple of (size_y, size_x) defining the size of the region of interest.
            as_NumPy (bool): If `True`, return the image as 4D `numpy.ndarray` object (c,z,y,x).
                Otherwise, return the (on-disk) `dask` array of the same dimensions.
        
        Returns:
            The extracted label mask, either as a `dask.array.Array` on-disk array, or as an in-memory `numpy.ndarray` if `as_NumPy=True`.
        
        Examples:
            Obtain the label mask of the lowest-resolution pyramid level for the full well 'A02':

            >>> plateA.get_label_ROI(well='A02')
        """
        # digest arguments
        well = self._digest_well_argument(well)
        assert label_name in self.label_names, f"Unknown label_name {label_name}, should be one of " + ', '.join(self.label_names)
        pyramid_level = self._digest_pyramid_level_argument(pyramid_level, ('image', image_name))

        # load image
        msk_path = os.path.join(self.path, well, image_name, 'labels', label_name, str(pyramid_level))
        msk = dask.array.from_zarr(msk_path)

        # calculate corner coordinates and subset if needed
        # (images are always of 4D shape c,z,y,x)
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
            if pyramid_level_coord != None and pyramid_level != pyramid_level_coord:
                upper_left_yx = self.convert_pixel_to_pixel(zyx=((0,) + upper_left_yx),
                                                            pyramid_level_from=pyramid_level_coord,
                                                            pyramid_level_to=pyramid_level,
                                                            pyramid_ref_from=('image', image_name),
                                                            pyramid_ref_to=('image', image_name))[1:]
                lower_right_yx = self.convert_pixel_to_pixel(zyx=((0,) + lower_right_yx),
                                                             pyramid_level_from=pyramid_level_coord,
                                                             pyramid_level_to=pyramid_level,
                                                             pyramid_ref_from=('image', image_name),
                                                             pyramid_ref_to=('image', image_name))[1:]
            msk = msk[:,
                      slice(upper_left_yx[0], lower_right_yx[0] + 1),
                      slice(upper_left_yx[1], lower_right_yx[1] + 1)]
        elif num_unknowns != 3:
            raise ValueError("Either none or two of `upper_left_yx`, `lower_right_yx` and `size_yx` have to be given")

        # convert if needed and return
        if as_NumPy:
            msk = np.array(msk)
        return msk

    def get_label_table_idx(self,
                            label_name: str,
                            table_name: str,
                            table_idx: int,
                            well: Optional[str]=None,
                            pyramid_level: Optional[int]=None,
                            as_NumPy: bool=False) -> Union[dask.array.Array, np.ndarray]:
        """
        Extract a region of interest from a label mask (segmentation) by table name and row index.

        Bounding box coordinates will be automatically obtained from the table
        `table_name` and row `row_idx` (zero-based row index).
        All z planes are returned if there are several ones.

        Parameters:
            label_name (str): The name of the segmentation
            table_name (str): The name of the table containing object coordinates in columns
                `x_micrometer`, `y_micrometer`, `len_x_micrometer` and `len_y_micrometer`.
            table_idx (int): The zero-based row index for the object to be extracted.
            well (str): The well (e.g. 'B03') from which an image should be extracted.
            pyramid_level (int): The pyramid level (resolution level), from which the image
                should be extracted. If `None`, the lowest-resolution (highest) pyramid level
                will be selected.
            as_NumPy (bool): If `True`, return the image as 4D `numpy.ndarray` object (c,z,y,x).
                Otherwise, return the (on-disk) `dask` array of the same dimensions.
        
        Returns:
            The extracted label mask, either as a `dask.array.Array` on-disk array, or as an in-memory `numpy.ndarray` if `as_NumPy=True`.
        
        Examples:
            Obtain the label mask in 'nuclei' of the first object in table `nuclei_ROI_table` in well 'A02':

            >>> plateA.get_label_table_idx(label_name='nuclei',
                                           table_name='nuclei_ROI_table',
                                           table_idx=0, well='A02')
        """
        # digest arguments
        assert label_name in self.label_names, f"Unknown label_name {label_name}, should be one of " + ', '.join(self.label_names)
        assert table_name in self.table_names, f"Unknown table_name {table_name}, should be one of " + ', '.join(self.table_names)
        well = self._digest_well_argument(well, as_path=False)
        pyramid_level = self._digest_pyramid_level_argument(pyramid_level)

        # extract table
        df = self.get_table(table_name=table_name, include_wells=well, as_AnnData=False)
        required_columns = ['x_micrometer', 'y_micrometer', 'len_x_micrometer', 'len_y_micrometer']
        assert all(column in set(df.columns) for column in required_columns), f"Missing columns: {set(required_columns) - set(df.columns)}"
        assert table_idx < len(df), f"table_idx ({table_idx}) needs to be less than " + str(len(df))

        # get bounding box coordinates
        ul = self.convert_micrometer_to_pixel((0,
                                               df['y_micrometer'].iloc[table_idx],
                                               df['x_micrometer'].iloc[table_idx]),
                                               pyramid_level=pyramid_level,
                                               pyramid_ref=('label', label_name))
        hw = self.convert_micrometer_to_pixel((0,
                                               df['len_y_micrometer'].iloc[table_idx],
                                               df['len_x_micrometer'].iloc[table_idx]),
                                               pyramid_level=pyramid_level,
                                               pyramid_ref=('label', label_name))


        # load image
        img = self.get_label_ROI(label_name=label_name,
                                 well=well,
                                 upper_left_yx=ul[1:],
                                 size_yx=hw[1:],
                                 pyramid_level=pyramid_level,
                                 as_NumPy=as_NumPy)

        return img

    def convert_micrometer_to_pixel(self,
                                    zyx: tuple[float],
                                    pyramid_level: Optional[int]=None,
                                    pyramid_ref: tuple[str]=('image', '0')) -> tuple[int]:
        """
        Convert micrometers to pixels for a given pyramid level.

        Parameters:
            zyx (tuple): The micrometer coordinates in the form (z, y, x) to be converted
                to pixels.
            pyramid_level (int): The pyramid level (resolution), which the output
                pixel coordinates will refer to.  If `None`, the lowest-resolution
                (highest) pyramid level will be selected.
            pyramid_ref (tuple(str, str)): The reference that the `pyramid_level` refers
                to. It is given as a tuple with two `str` elements, the first being
                either 'image' or 'label', and the second being the name of the
                image or label. The default is ('image', '0').
        
        Returns:
            A tuple (z, y, x) with pixel coordinates.
        
        Examples:
            Obtain the x-length (x_px) of a scale bar corresopnding to 10 micrometer for
            pyramid level 0:

            >>> z_px, y_px, x_px = plateA.convert_micrometer_to_pixel(zyx=(0,0,10), pyramid_level=0)
        """
        # digest arguments
        assert isinstance(zyx, tuple), "zyx needs to be a tuple of the form (z,y,x)"
        pyramid_level = self._digest_pyramid_level_argument(pyramid_level)

        # convert
        if pyramid_ref[0] == 'image':
            pyramid_spacing = self.level_zyx_spacing_images[pyramid_ref[1]][pyramid_level]
        else:
            pyramid_spacing = self.level_zyx_spacing_labels[pyramid_ref[1]][pyramid_level]
        zyx_px = np.round(np.divide(zyx, pyramid_spacing)).astype(int)

        # return as tuple
        return(tuple(zyx_px))

    def convert_pixel_to_micrometer(self,
                                    zyx: tuple[int],
                                    pyramid_level: Optional[int]=None,
                                    pyramid_ref: tuple[str]=('image', '0')) -> tuple[float]:
        """
        Convert pixels to micrometers for a given pyramid level.

        Parameters:
            zyx (tuple): The pixel coordinates in the form (z, y, x) to be converted
                to micrometers.
            pyramid_level (int): The pyramid level (resolution), which the input
                pixel coordinates refer to. If `None`, the lowest-resolution
                (highest) pyramid level will be selected.
            pyramid_ref (tuple(str, str)): The reference that the `pyramid_level` refers
                to. It is given as a tuple with two `str` elements, the first being
                either 'image' or 'label', and the second being the name of the
                image or label. The default is ('image', '0').
        
        Returns:
            A tuple (z, y, x) with micrometer coordinates.
        
        Examples:
            Obtain the micrometer dimensions of a cube with 10 pixel sides for pyramid level 0:

            >>> z_um, y_um, x_um = plateA.convert_pixel_to_micrometer(zyx=(10,10,10), pyramid_level=0)
        """
        # digest arguments
        pyramid_level = self._digest_pyramid_level_argument(pyramid_level)

        # convert
        if pyramid_ref[0] == 'image':
            pyramid_spacing = self.level_zyx_spacing_images[pyramid_ref[1]][pyramid_level]
        else:
            pyramid_spacing = self.level_zyx_spacing_labels[pyramid_ref[1]][pyramid_level]
        zyx_um = np.round(np.multiply(zyx, pyramid_spacing)).astype(float)

        # return as tuple
        return(tuple(zyx_um))

    def convert_pixel_to_pixel(self,
                               zyx: tuple[int],
                               pyramid_level_from: Optional[int]=None,
                               pyramid_level_to: Optional[int]=None,
                               pyramid_ref_from: tuple[str]=('image', '0'),
                               pyramid_ref_to: tuple[str]=('image', '0')) -> tuple[int]:
        """
        Convert pixel coordinates between pyramid levels.

        Parameters:
            zyx (tuple): The pixel coordinates in the form (z, y, x) to be converted.
            pyramid_level_from (int): The pyramid level (resolution), which the input
                pixel coordinates refer to. If `None`, the lowest-resolution
                (highest) pyramid level will be selected.
            pyramid_level_to (int): The pyramid level (resolution), which the output
                pixel coordinates will refer to. If `None`, the lowest-resolution
                (highest) pyramid level will be selected.
            pyramid_ref_from (tuple(str, str)): The reference that the `pyramid_level_from` refers
                to. It is given as a tuple with two `str` elements, the first being
                either 'image' or 'label', and the second being the name of the
                image or label. The default is ('image', '0').
            pyramid_ref_to (tuple(str, str)): The reference that the `pyramid_level_to` refers
                to. It is given as a tuple with two `str` elements, the first being
                either 'image' or 'label', and the second being the name of the
                image or label. The default is ('image', '0').
        
        Returns:
            A tuple (z, y, x) with pixel coordinates in the new pyramid level.
        
        Examples:
            Convert a point (0, 10, 30) from pyramid level 3 to 0:

            >>> z0, y0, x0 = plateA.convert_pixel_to_pixel(zyx=(0,10,30), pyramid_level_from=3, pyramid_level_to=0)
        """
        # digest arguments
        pyramid_level_from = self._digest_pyramid_level_argument(pyramid_level_from)
        pyramid_level_to = self._digest_pyramid_level_argument(pyramid_level_to)

        # convert
        if pyramid_ref_from[0] == 'image':
            pyramid_spacing_from = self.level_zyx_spacing_images[pyramid_ref_from[1]][pyramid_level_from]
        else:
            pyramid_spacing_from = self.level_zyx_spacing_labels[pyramid_ref_from[1]][pyramid_level_from]
        if pyramid_ref_to[0] == 'image':
            pyramid_spacing_to = self.level_zyx_spacing_images[pyramid_ref_to[1]][pyramid_level_to]
        else:
            pyramid_spacing_to = self.level_zyx_spacing_labels[pyramid_ref_to[1]][pyramid_level_to]

        zyx_scale = np.divide(pyramid_spacing_from, pyramid_spacing_to)
        zyx_new = np.round(np.multiply(zyx, zyx_scale)).astype(int)

        # return as tuple
        return(tuple(zyx_new))


    # analysis methods --------------------------------------------------------
    def calc_average_FOV(self,
                         include_wells: Union[str, list[str]]=[],
                         pyramid_level: Optional[int]=None,
                         channel: int=0) -> np.ndarray:
        """
        Calculate the average field of view.
         
        Using the coordinates stored in table 'FOV_ROI_table', calculate the averaged
        field of view for wells in `include_wells`, for `channel` at resolution `pyramid_level`.

        Parameters:
            include_wells (str or list): List of well names to include. If empty `[]`, all wells are included.
            pyramid_level (int): The pyramid level (resolution level), from which the image
                should be extracted. If `None`, the lowest-resolution (highest) pyramid level
                will be selected.
            channel (int): The channel for which the fields of view should be averaged.

        Returns:
            The averaged field of view, as an array of shape (z,y,x).

        Examples:
            Calculate the averaged field of view for channel zero over all wells
            for pyramid level 1.

            >>> avg_fov = plateA.calc_average_FOV(pyramid_level=1, channel=0)
        """
        # check if required data is available
        if not 'FOV_ROI_table' in self.table_names:
            raise ValueError("`FOV_ROI_table` not found - cannot calculate average FOV")
        
        # digest arguments
        include_wells = self._digest_include_wells_argument(include_wells)
        pyramid_level = self._digest_pyramid_level_argument(pyramid_level)

        # extract FOV table and scaling information
        fov_tab = self.get_table('FOV_ROI_table')
        pyramid_spacing = self.level_zyx_spacing_images['0'][pyramid_level]
        assert len(pyramid_spacing) == 3
        pyramid_spacing = pyramid_spacing[1:] # scale is (z, y, x), just keep y, x

        # sum FOVs
        avg_fov = np.zeros((0))
        n = 0
        for well, group in fov_tab.groupby('well'):
            well = self._digest_well_argument(well)
            if well in include_wells:

                # load full well image
                img_path = os.path.join(self.path, well, '0', str(pyramid_level))
                img = dask.array.from_zarr(img_path)
                
                # calculate coordinates for `pyramid_level` that correspond to fields of view
                fov_yx_start_um = group[['y_micrometer', 'x_micrometer']].values
                fov_yx_end_um = fov_yx_start_um + group[['len_y_micrometer', 'len_x_micrometer']].values
                fov_yx_start_px = np.round(np.divide(fov_yx_start_um, pyramid_spacing)).astype(int)
                fov_yx_end_px = np.round(np.divide(fov_yx_end_um, pyramid_spacing)).astype(int)
                if len(avg_fov) == 0:
                    avg_fov = np.zeros((img.shape[1],
                                        fov_yx_end_px[0,0] - fov_yx_start_px[0,0],
                                        fov_yx_end_px[0,1] - fov_yx_start_px[0,1]))

                # add fields of view to `avg_fov`
                for i in range(group.shape[0]):
                    fov_img = img[:,
                                  :,
                                  slice(fov_yx_start_px[i,0], fov_yx_end_px[i,0]),
                                  slice(fov_yx_start_px[i,1], fov_yx_end_px[i,1])]
                    n += 1
                    avg_fov += fov_img[channel].compute()
        
        # calculate mean
        avg_fov /= n
        return(avg_fov)

    # plotting methods -----------------------------------------------------------
    def plot_well(self,
                  well: str,
                  upper_left_yx: Optional[tuple[int]]=None,
                  lower_right_yx: Optional[tuple[int]]=None,
                  size_yx: Optional[tuple[int]]=None,
                  image_name: str='0',
                  label_name: Optional[str]=None,
                  label_alpha: float=0.3,
                  pyramid_level: Optional[int]=None,
                  pyramid_level_coord: Optional[int]=None,
                  channels: list[int]=[0],
                  channel_colors: list[str]=['white'],
                  channel_ranges: list[list[float]]=[[0.01, 0.95]],
                  z_projection_method: str='maximum',
                  axis_style: str='pixel',
                  title: Optional[str]=None,
                  scalebar_micrometer: int=0,
                  scalebar_color: str='white',
                  scalebar_position: str='bottomright',
                  scalebar_label: bool=False,
                  call_show: bool=True,
                  fig_width_inch: float=8.0,
                  fig_height_inch: float=8.0,
                  fig_dpi: int=200,
                  fig_style: str='dark_background'):
        """
        Plot a well from a microtiter plate.
         
        Plot an overview of a single well, for `channel` at
        resolution `pyramid_level`.

        Parameters:
            well (str): The well (e.g. 'B03') to be plotted.
            upper_left_yx (tuple): Tuple of (y, x) coordinates for the upper-left
                (lower) coordinates defining the region of interest.
            lower_right_yx (tuple): Tuple of (y, x) coordinates for the lower-right
                (higher) coordinates defining the region of interest.
            size_yx (tuple): Tuple of (size_y, size_x) defining the size of the
                region of interest.
            image_name (str): The name of the image in the well to be plotted.
                Default: '0'.
            label_name (str): The name of the a segmentation mask to be plotted
                semi-transparently over the images. If `None`, just the image
                is plotted.
            label_alpha (float): A scalar value between 0 (fully transparent)
                and 1 (solid) defining the transparency of the label masks.
            pyramid_level (int): The pyramid level (resolution level), from
                which the image should be extracted. If `None`, the
                lowest-resolution (highest) pyramid level will be selected.
            pyramid_level_coord (int): An optional integer scalar giving the image pyramid level
                to which the coordinates (`upper_left_yx`, `lower_right_yx` and `size_yx`)
                refer to. By default, this is `None`, which will use `pyramid_level`.
            channels (list[int]): The image channel(s) to be plotted.
            channel_colors (list[str]): A list with python color strings
                (e.g. 'red') defining the color for each channel in `channels`.
            channel_ranges (list[list[float]]): A list of 2-element lists
                (e.g. [0.01, 0.95]) giving the value ranges that should be
                mapped to colors for each channel. If the given numerical values
                are less or equal to 1.0, they are interpreted as quantiles and
                the corresponding intensity values are calculated on the channel
                non-zero values, otherwise they are directly used as intensities.
                Values outside of this range will be clipped.
            z_projection_method (str): Method for combining multiple z planes.
                For available methods, see ez_zarr.plotting.zproject
                (default: 'maximum').
            axis_style (str): A string scalar defining how to draw the axis. Should
                be one of 'none' (no axis), 'pixel' (show pixel labels, the default)
                or 'micrometer' (show micrometer labels). If `axis_style='micrometer'`,
                `spacing_yx` is used to convert pixel to micrometer.
            title (str): String scalar to add as title. If `None`, `well` will
                be used as `title`.
            scalebar_micrometer (int): If non-zero, add a scale bar corresponding
                to `scalebar_micrometer` to the bottom right.
            scalebar_label (bool): If `True`, add micrometer label to scale bar.
            call_show (bool): If `True`, the call to `matplotlib.pyplot.imshow` is
                embedded between `matplotlib.pyplot.figure` and
                `matplotlib.pyplot.show`/`matplotlib.pyplot.close` calls.
                This is the default behaviour and typically used when an individual
                image should be plotted and displayed. It can be set to `False`
                if multiple images should be plotted and their arrangement
                is controlled outside of `plotting.plot_image`. The parameters
                `fig_width_inch`, `fig_height_inch` and `fig_dpi` are ignored
                in that case.
            fig_width_inch (float): Figure width (in inches).
            fig_height_inch (float): Figure height (in inches).
            fig_dpi (int): Figure resolution (dots per inch).
            fig_style (str): Style for the figure. Supported are 'brightfield', which
                is a special mode for single-channel brightfield microscopic images
                (it will automatically set `channels=[0]`, `channel_colors=['white']`
                `z_projection_method='minimum'` and `fig_style='default'`), and any
                other styles that can be passed to `matplotlib.pyplot.style.context`
                (default: 'dark_background')

        Examples:
            Overview plot of the well 'B03'.

            >>> plateA.plot_well(well='B03')
        """
        # digest arguments
        well = self._digest_well_argument(well, as_path=False)
        assert image_name in self.image_names, (
            f"Unknown image_name ({image_name}), should be one of "
            ', '.join(self.image_names)
        )
        assert label_name == None or label_name in self.label_names, (
            f"Unknown label_name ({label_name}), should be `None` or one of "
            ', '.join(self.label_names)
        )
        img_pl = self._digest_pyramid_level_argument(pyramid_level=pyramid_level,
                                                     pyramid_ref=('image', image_name))
        assert all(ch < len(self.channels) for ch in channels), (
            f"Invalid channels ({channels}), must be less than {len(self.channels)}"
        )

        # import optional modules
        plotting = importlib.import_module('ez_zarr.plotting')
        
        # get mask pyramid level corresponding to `img_pl`
        if label_name != None:
            pl_match = [self.level_zyx_spacing_images[image_name][img_pl] == x
                        for x in self.level_zyx_spacing_labels[label_name]]
            assert sum(pl_match) == 1, (
                f"Could not find a label pyramid level corresponding to the "
                f"selected image pyramid level ({img_pl})"
            )
            msk_pl = pl_match.index(True)

        # get well image
        img = self.get_image_ROI(well=well,
                                 upper_left_yx=upper_left_yx,
                                 lower_right_yx=lower_right_yx,
                                 size_yx=size_yx,
                                 pyramid_level=img_pl,
                                 pyramid_level_coord=pyramid_level_coord,
                                 as_NumPy=True)

        if label_name != None:
            msk = self.get_label_ROI(label_name=label_name,
                                     well=well,
                                     upper_left_yx=upper_left_yx,
                                     lower_right_yx=lower_right_yx,
                                     size_yx=size_yx,
                                     pyramid_level=msk_pl,
                                     pyramid_level_coord=pyramid_level_coord,
                                     as_NumPy=True)
            assert img.shape[1:] == msk.shape, (
                f"label {label_name} shape {msk.shape} does not match "
                f"image shape {img.shape} for well {well}"
            )
        else:
            msk = None

        # calculate scalebar length in pixel in x direction
        if scalebar_micrometer != 0:
            scalebar_pixel = self.convert_micrometer_to_pixel(zyx = (0, 0, scalebar_micrometer),
                                                               pyramid_level=img_pl)[2]
        else:
            scalebar_pixel = 0

        # plot well
        if title is None:
            title = well
        if scalebar_label:
            scalebar_label = str(scalebar_micrometer) + ' µm'
        else:
            scalebar_label = None
        plotting.plot_image(im=img,
                            msk=msk,
                            msk_alpha=label_alpha,
                            channels=channels,
                            channel_colors=channel_colors,
                            channel_ranges=channel_ranges,
                            z_projection_method=z_projection_method,
                            axis_style=axis_style,
                            spacing_yx=self.level_zyx_spacing_images[image_name][img_pl][1:],
                            title=title,
                            scalebar_pixel=scalebar_pixel,
                            scalebar_color=scalebar_color,
                            scalebar_position=scalebar_position,
                            scalebar_label=scalebar_label,
                            call_show=call_show,
                            fig_width_inch=fig_width_inch,
                            fig_height_inch=fig_height_inch,
                            fig_dpi=fig_dpi,
                            fig_style=fig_style)

    def plot_plate(self,
                   image_name: str='0',
                   label_name: Optional[str]=None,
                   label_alpha: float=0.3,
                   pyramid_level: Optional[int]=None,
                   channels: list[int]=[0],
                   channel_colors: list[str]=['white'],
                   channel_ranges: list[list[float]]=[[0.01, 0.95]],
                   z_projection_method: str='maximum',
                   plate_layout: str='96well',
                   fig_title: Optional[str]=None,
                   fig_width_inch: float=24.0,
                   fig_height_inch: float=16.0,
                   fig_dpi: int=200,
                   fig_style: str='dark_background'):
        """
        Plot microtiter plate.
         
        Plot an overview of all wells in plate arrangement, for `channel` at
        resolution `pyramid_level`.

        Parameters:
            image_name (str): The name of the image in each well to be plotted.
                Default: '0'.
            label_name (str): The name of the segmentation mask to be plotted
                semi-transparently over the images. If `None`, just the image
                is plotted.
            label_alpha (float): A scalar value between 0 (fully transparent)
                and 1 (solid) defining the transparency of the label masks.
            pyramid_level (int): The pyramid level (resolution level), from
                which the image should be extracted. If `None`, the
                lowest-resolution (highest) pyramid level will be selected.
            channels (list[int]): The image channel(s) to be plotted.
            channel_colors (list[str]): A list with python color strings
                (e.g. 'red') defining the color for each channel in `channels`.
            channel_ranges (list[list[float]]): A list of 2-element lists
                (e.g. [0.01, 0.95]) giving the value ranges that should be
                mapped to colors for each channel. If the given numerical values
                are less or equal to 1.0, they are interpreted as quantiles and
                the corresponding intensity values are calculated on the channel
                non-zero values, otherwise they are directly used as intensities.
                Values outside of this range will be clipped.
            z_projection_method (str): Method for combining multiple z planes.
                For available methods, see ez_zarr.plotting.zproject
                (default: 'maximum').
            plate_layout (str): Defines the layout of the plate
                (default: '96well').
            fig_title (str): String scalar to use as overall figure title
                (default: the `.name` attribute of the object). Use `fig_title=''`
                to not add any title to the plot.
            fig_width_inch (float): Figure width (in inches).
            fig_height_inch (float): Figure height (in inches).
            fig_dpi (int): Figure resolution (dots per inch).
            fig_style (str): Style for the figure. Supported are 'brightfield', which
                is a special mode for single-channel brightfield microscopic images
                (it will automatically set `channels=[0]`, `channel_colors=['white']`
                `z_projection_method='minimum'` and `fig_style='default'`), and any
                other styles that can be passed to `matplotlib.pyplot.style.context`
                (default: 'dark_background')

        Examples:
            Overview plot of a plate for image channel 1.

            >>> plateA.plot_plate(channels=[1])
        """
        # digest arguments
        assert image_name in self.image_names, (
            f"Unknown image_name ({image_name}), should be one of "
            ', '.join(self.image_names)
        )
        assert label_name == None or label_name in self.label_names, (
            f"Unknown label_name ({label_name}), should be `None` or one of "
            ', '.join(self.label_names)
        )
        img_pl = self._digest_pyramid_level_argument(pyramid_level=pyramid_level,
                                                     pyramid_ref=('image', image_name))
        assert all(ch < len(self.channels) for ch in channels), (
            f"Invalid channels ({channels}), must be less than {len(self.channels)}"
        )

        # import optional modules
        plotting = importlib.import_module('ez_zarr.plotting')
        assert plate_layout in plotting.plate_layouts, (
            f"Unknown plate_layout ({plate_layout}), should be one of "
            ', '.join(list(plotting.plate_layouts.keys()))
        )
        plt = importlib.import_module('matplotlib.pyplot')
        
        # plate layout
        rows = plotting.plate_layouts[plate_layout]['rows']
        columns = plotting.plate_layouts[plate_layout]['columns']

        # available wells
        wells = self.get_wells(simplify=True)

        # get mask pyramid level corresponding to `img_pl`
        if label_name != None:
            pl_match = [self.level_zyx_spacing_images[image_name][img_pl] == x
                        for x in self.level_zyx_spacing_labels[label_name]]
            assert sum(pl_match) == 1, (
                f"Could not find a label pyramid level corresponding to the "
                f"selected image pyramid level ({img_pl})"
            )
            msk_pl = pl_match.index(True)

        # get the maximal well y,x coordinates
        well_dims = [self.get_image_ROI(well=w, pyramid_level=img_pl).shape[2:] for w in wells]
        max_yx = np.max(np.stack(well_dims), axis=0)

        # adjust parameters for brightfield images
        if fig_style == 'brightfield':
            channels = [0]
            channel_colors = ['white']
            z_projection_method = 'minimum'
            fig_style = 'default'
            empty_well = np.ones(max_yx)
        else:
            empty_well = np.zeros(max_yx)

        # define figure title
        if fig_title is None:
            fig_title = self.name

        # loop over wells
        with plt.style.context(fig_style):
            fig = plt.figure(figsize=(fig_width_inch, fig_height_inch))
            fig.set_dpi(fig_dpi)
            if fig_title != '':
                fig.suptitle(fig_title, size='xx-large') # default title size: 'large'
            for r in range(len(rows)):
                for c in range(len(columns)):
                    w = rows[r] + columns[c]
                    plt.subplot(len(rows), len(columns), r * len(columns) + c + 1)

                    if w in wells:
                        # get well image
                        img = self.get_image_ROI(well=w,
                                                 pyramid_level=img_pl,
                                                 as_NumPy=True)
                        img_shape_before_padding = img.shape

                        # add segmentation mask on top
                        if label_name != None:
                            # get well label
                            msk = self.get_label_ROI(label_name=label_name,
                                                     well=w,
                                                     pyramid_level=msk_pl,
                                                     as_NumPy=True)
                            assert img_shape_before_padding[1:] == msk.shape, (
                                f"label {label_name} shape {msk.shape} does not match "
                                f"image shape {img_shape_before_padding} for well {w}"
                            )
                        else:
                            msk = None

                        # plot well
                        plotting.plot_image(im=img,
                                            msk=msk,
                                            msk_alpha=label_alpha,
                                            channels=channels,
                                            channel_colors=channel_colors,
                                            channel_ranges=channel_ranges,
                                            z_projection_method=z_projection_method,
                                            pad_to_yx=max_yx,
                                            axis_style='frame',
                                            title=w,
                                            call_show=False)
                    else:
                        # plot empty well
                        plt.imshow(empty_well, cmap='gray', vmin=0, vmax=1)
                        plt.xticks([]) # remove axis ticks
                        plt.yticks([])
                        plt.title(w)
            fig.tight_layout()
            plt.show()
            plt.close()






# FractalZarrSet class ------------------------------------------------------
class FractalZarrSet:
    """Represents a folder containing one or several ome-zarr fileset(s)."""

    # constructor and helper functions ----------------------------------------
    def __init__(self, path: str, name=None) -> None:
        """
        Initializes a container for a folder containing one or several OME-Zarr
        fileset(s) (.zarr). Typically, the object is used for a folder which
        contains exactly two related `.zarr` objects, one corresponding to the
        four-dimensional (c,z,y,x) plate dataset, and a second one corresponding to
        a three-dimensional maximum intensity projection derived from it.

        Parameters:
            path (str): Path containing the OME-Zarr fileset(s).
            name (str): Optional name for the experiment.
        
        Examples:
            Get an object corresponding to a set of `.zarr`s.

            >>> from ez_zarr import hcs_wrappers
            >>> plate_set = hcs_wrappers.FractalZarrSet('path/to/zarrs')
            >>> plate_set

            This will print information on the `.zarr`s.
        """
        if not os.path.isdir(path):
            raise ValueError(f'`{path}` does not exist')
        self.path: str = path
        self.name: str = ''
        if name is None:
            self.name = os.path.basename(self.path)
        else:
            self.name = name
        self.zarr_paths: list[str] = [f for f in os.listdir(self.path) if f[-5:] == '.zarr']
        if len(self.zarr_paths) == 0:
            raise ValueError(f'no `.zarr` filesets found in `{path}`')
        self.zarr_paths.sort(reverse=True) # defined order (_mip.zarr before .zarr)
        self.zarr_mip_idx: Optional[int] = None
        self.zarr_3d_idx: Optional[int] = None
        if len(self.zarr_paths) == 2:
            if self.zarr_paths[0].replace('_mip.zarr', '.zarr') == self.zarr_paths[1]:
                # special case of 3D plate plus derived maximum intensity projection?
                self.zarr_mip_idx = 0
                self.zarr_3d_idx = 1
        self.zarr: list[FractalZarr] = [FractalZarr(os.path.join(self.path, f)) for f in self.zarr_paths]
        self.zarr_names: list[str] = [x.name for x in self.zarr]

    # string representation ---------------------------------------------------
    def __str__(self) -> str:
        nplates = len(self.zarr)
        platenames = ''.join(f'    {i}: {self.zarr[i].name}\n' for i in range(len(self.zarr)))
        return f"FractalZarrSet {self.name}\n  path: {self.path}\n  n_plates: {nplates}\n{platenames}\n"
    
    def __repr__(self) -> str:
        return str(self)

    # accessors ---------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.zarr)

    def __getitem__(self, key: Union[int, str]) -> FractalZarr:
        if isinstance(key, int):
            return self.zarr[key]
        elif isinstance(key, str):
            return self.zarr[self.zarr_names.index(key)]

    def __getattr__(self, name):
        try:
            attr = getattr(self.zarr[0], name)
        except AttributeError:
            raise AttributeError(f"'{type(self.zarr[0]).__name__}' objects have no attribute '{name}'")

        if callable(attr):
            def wrapper(*args, **kwargs):
                return [getattr(plate, name)(*args, **kwargs) for plate in self.zarr]
            return wrapper
        else:
            return [getattr(plate, name) for plate in self.zarr]
