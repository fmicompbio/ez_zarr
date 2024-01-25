"""Wrap ome-zarr filesets in a class.

Represent an ome-zarr fileset as a class to give high-level
access to its contents.

Classes:
    FractalZarr: Contains a single .zarr fileset, typically a plate.
    FractalZarrSet: Contains one or several .zarr filesets, typically a plate
        (4 dimensional data) and a maximum-intensity projection derived from it.
"""

__all__ = ['FractalZarr', 'FractalZarrSet']
__version__ = '0.1.1'
__author__ = 'Silvia Barbiero, Michael Stadler'


# imports ---------------------------------------------------------------------
import os, re
import numpy as np
import zarr
import dask
import anndata as ad
import pandas as pd
# import tifffile
import warnings
import random
from typing import Union, Optional, Any


# FractalZarr class ---------------------------------------------------------------
class FractalZarr:
    """Represents a ome-zarr fileset."""

    # constructor and helper functions ----------------------------------------
    def __init__(self, zarr_path: str, name: Optional[str] = None) -> None:
        """
        Initializes an ome-zarr fileset (.zarr) from its path.
        Typically, the fileset represents a single assay plate, and 
        we assume that the structures (pyramid levels, labels, table, etc.)
        are consistent across wells.

        Parameters:
            zarr_path (str): Path containing the plate ome-zarr fileset.
            name (str, optional): Optional name for the plate.
        
        Examples:
            Get an object corresponding to a plate.

            >>> from ez_zarr import hcs_wrappers
            >>> plateA = hcs_wrappers.FractalZarr('path/to/plate.zarr')
            >>> plateA

            This will print information on the plate.
        """

        self.path: str = zarr_path
        self.name: str = ''
        if name:
            self.name = name
        else:
            self.name = os.path.basename(self.path)
        self.__top: zarr.Group = zarr.open_group(store = self.path, mode = 'r')
        if not 'plate' in self.__top.attrs:
            raise ValueError(f"{self.name} does not contain a zarr fileset with a 'plate'")
        self.acquisitions: list = self.__top.attrs['plate']['acquisitions']
        self.columns: list[dict] = self.__top.attrs['plate']['columns']
        self.rows: list[dict] = self.__top.attrs['plate']['rows']
        self.wells: list[dict] = self.__top.attrs['plate']['wells']
        self.channels: list[dict] = self._load_channel_info()
        self.multiscales: dict[str, Any] = self._load_multiscale_info()
        self.level_paths: list[str] = [x['path'] for x in self.multiscales['datasets']]
        self.level_zyx_spacing: list[list[float]] = [x['coordinateTransformations'][0]['scale'][1:] for x in self.multiscales['datasets']] # convention: unit is micrometer
        self.level_zyx_scalefactor: np.ndarray = np.divide(self.level_zyx_spacing[1], self.level_zyx_spacing[0])
        self.label_names: list[str] = self._load_label_names()
        self.table_names: list[str] = self._load_table_names()

    def _load_channel_info(self) -> list:
        """[internal] Load info about available channels."""
        well = self.wells[0]['path']
        well_group = self.__top[os.path.join(well, '0')] # convention: single field of view per well
        if not 'omero' in well_group.attrs or not 'channels' in well_group.attrs['omero']:
            raise ValueError(f"no channel info found in well {well}")
        return well_group.attrs['omero']['channels']
    
    def _load_multiscale_info(self) -> dict[str, Any]:
        """[internal] Load info about available scales."""
        well = self.wells[0]['path']
        well_group = self.__top[os.path.join(well, '0')] # convention: single field of view per well
        if not 'multiscales' in well_group.attrs:
            raise ValueError(f"no multiscale info found in well {well}")
        return well_group.attrs['multiscales'][0]
    
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
    def _digest_well_argument(self, well = None):
        """[internal] Interpret a single `well` argument in the context of a given FractalZarr object."""
        if not well:
            # no well given -> pick first one
            return self.wells[0]['path']
        else:
            return os.path.join(well[:1].upper(), well[1:])

    def _digest_include_wells_argument(self, include_wells: list[str] = []) -> list[str]:
        """[internal] Interpret an `include_wells` argument in the context of a given FractalZarr object."""
        if len(include_wells) == 0: 
            # no wells given -> include all wells
            include_wells = [x['path'] for x in self.wells]
        else:
            # transform well names from 'B03' format to path format 'B/03'
            include_wells = [self._digest_well_argument(w) for w in include_wells]
        return include_wells

    def _digest_pyramid_level_argument(self, pyramid_level = None) -> int:
        """[internal] Interpret a `pyramid_level` argument in the context of a given FractalZarr object."""
        if pyramid_level == None: 
            # no pyramid level given -> pick lowest resolution one
            pyramid_level = int(self.level_paths[-1])
        else:
            # make sure it is an integer
            pyramid_level = int(pyramid_level)
        return pyramid_level
    
    def _calculate_regular_grid_coordsinates(self, y, x, num_y = 10, num_x = 10):
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
        npl = len(self.multiscales['datasets'])
        segnames = ', '.join(self.label_names)
        tabnames = ', '.join(self.table_names)
        return f"FractalZarr {self.name}\n  path: {self.path}\n  n_wells: {nwells}\n  n_channels: {nch} ({chlabs})\n  n_pyramid_levels: {npl}\n  pyramid_zyx_scalefactor: {self.level_zyx_scalefactor}\n  full_resolution_zyx_spacing: {self.level_zyx_spacing[0]}\n  segmentations: {segnames}\n  tables (measurements): {tabnames}\n"
    
    def __repr__(self):
        return str(self)
    
    # accessors ---------------------------------------------------------------
    def get_path(self) -> str:
        """Gets the path of the ome-zarr fileset.
        
        Returns:
            The path to the ome-zarr fileset.
        """
        return self.path

    def get_wells(self, simplify: bool = False) -> Union[list[dict[str, Any]], list[str]]:
        """Gets info on wells in the ome-zarr fileset.

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
        """Gets info on channels in the ome-zarr fileset.
        
        Returns:
            A list of dicts with information on channels.
        """
        return self.channels
    
    def get_table_names(self) -> list:
        """Gets list of table names in the ome-zarr fileset.
        
        Returns:
            A list of table names (str) available in the plate.
        """
        return self.table_names

    # query methods -----------------------------------------------------------
    def get_table(self,
                  table_name: str,
                  include_wells: list[str] = [],
                  as_AnnData: bool = False) -> Union[ad.AnnData, pd.DataFrame]:
        """Extract table for wells in a ome-zarr fileset.
        
        Parameters:
            table_name (str): The name of the table to extract.
            include_wells (list): List of well names to include. If empty `[]`, all wells are included.
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
                                     axis = 1)
            return df_combined

    def get_image_ROI(self,
                      well: Optional[str] = None,
                      pyramid_level: Optional[int] = None,
                      upper_left_yx: Optional[tuple[int]] = None,
                      lower_right_yx: Optional[tuple[int]] = None,
                      size_yx: Optional[tuple[int]] = None,
                      as_NumPy: bool = False) -> Union[dask.array.Array, np.ndarray]:
        """
        Extract a region of interest from a well image by coordinates.

        None or at least two of `upper_left_yx`, `lower_right_yx` and `size_yx` need to be given.
        If none are given, it will return the full image (the whole well).
        Otherwise, `upper_left_yx` contains the lower indices than `lower_right_yx`
        (origin on the top-left, zero-based coordinates), and each of them is
        a tuple of (y, x). No z coordinate needs to be given, all z planes are returned
        if there are several ones.

        Parameters:
            well (str): The well (e.g. 'B03') from which an image should be extracted.
            pyramid_level (int): The pyramid level (resolution level), from which the image
                should be extracted. If `None`, the lowest-resolution (highest) pyramid level
                will be selected.
            upper_left_yx (tuple): Tuple of (y, x) coordinates for the upper-left (lower) coordinates
                defining the region of interest.
            lower_right_yx (tuple): Tuple of (y, x) coordinates for the lower-right (higher) coordinates defining the region of interest.
            size_yx (tuple): Tuple of (size_y, size_x) defining the size of the region of interest.
            as_NumPy (bool): If `True`, return the image as 4D `numpy.ndarray` object (c,z,y,x).
                Otherwise, return the (on-disk) `dask` array of the same dimensions.
        
        Returns:
            The extracted image, either as a `dask.array.Array` on-disk array, or as an in-memory `numpy.ndarray` if `as_NumPy=True`.
        
        Examples:
            Obtain the image of the lowest-resolution for the full well 'A02':

            >>> plateA.get_image_ROI(well = 'A02')
        """
        # digest arguments
        well = self._digest_well_argument(well)
        pyramid_level = self._digest_pyramid_level_argument(pyramid_level)

        # load image (convention: single field of view per well -> '0')
        img_path = os.path.join(self.path, well, '0', str(pyramid_level))
        img = dask.array.from_zarr(img_path)

        # calculate corner coordinates and subset if needed
        # (images are always of 4D shape c,z,y,x)
        num_unknowns = sum([x == None for x in [upper_left_yx, lower_right_yx, size_yx]])
        if num_unknowns == 1:
            if size_yx:
                assert all([x > 0 for x in size_yx])
                if not upper_left_yx:
                    upper_left_yx = tuple(lower_right_yx[i] - size_yx[i] for i in range(2))
                elif not lower_right_yx:
                    lower_right_yx = tuple(upper_left_yx[i] + size_yx[i] for i in range(2))
            assert all([upper_left_yx[i] < lower_right_yx[i] for i in range(len(upper_left_yx))])
            img = img[:,
                      :,
                      slice(upper_left_yx[0], lower_right_yx[0] + 1),
                      slice(upper_left_yx[1], lower_right_yx[1] + 1)]
        elif num_unknowns != 3:
            raise ValueError("Either none or two of `upper_left_yx`, `lower_rigth` and `size_yx` have to be given")

        # convert if needed and return
        if as_NumPy:
            img = np.array(img)
        return img

    def get_image_grid_ROIs(self,
                            well: Optional[str] = None,
                            pyramid_level: Optional[int] = None,
                            lowres_level: Optional[int] = None,
                            num_y: int = 10,
                            num_x: int = 10,
                            num_select: int = 9,
                            sample_method: str = 'sum',
                            channel: int = 0,
                            seed: int = 42,
                            as_NumPy: bool = False) -> Union[tuple[list[tuple[int]], list[dask.array.Array]], tuple[list[tuple[int]], list[np.ndarray]]]:
        """
        Split a well image into a regular grid and extract a subset of grid cells (all z planes if several).

        `num_y` and `num_x` define the grid by specifying the number of cell in y and x.
        `num_select` picks that many from the total number of grid cells and returns them as a list.
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
            seed (int): Used in `random.seed()` to make sampling for `sample_method = 'random'` reproducible.
            as_NumPy (bool): If `True`, return the grid cell image as `numpy.ndarray` objects with
                shapes (c,z,y,x). Otherwise, return the (on-disk) `dask` arrays of the same dimensions.
        
        Returns:
            A tuple of two lists (coord_list, img_list), each with `num_select` elements
            corresponding to the coordinates and images of selected grid cells.
            The coordinates are tuples of the form  (y_start, y_end, x_start, x_end).

        Examples:
            Obtain grid cells with highest signal sum in channel 0 from well 'A02':

            >>> plateA.get_image_grid_ROIs(well = 'A02')
        """
        # digest arguments
        well = self._digest_well_argument(well)
        pyramid_level = self._digest_pyramid_level_argument(pyramid_level)
        lowres_level = self._digest_pyramid_level_argument(lowres_level)
        assert num_select <= num_x * num_y

        # load image (convention: single field of view per well -> '0')
        img_path = os.path.join(self.path, well, '0', str(pyramid_level))
        img = dask.array.from_zarr(img_path)
        img_path_lowres = os.path.join(self.path, well, '0', str(lowres_level))
        img_lowres = dask.array.from_zarr(img_path_lowres)

        # calculate grid coordinates as a list of (y_start, y_end, x_start, x_end)
        # (images are always of 4D shape c,z,y,x)
        ch, z, y, x = img.shape
        grid = self._calculate_regular_grid_coordsinates(y = y, x = x,
                                                         num_y = num_y,
                                                         num_x = num_x)
        ch_lr, z_lr, y_lr, x_lr = img_lowres.shape
        grid_lowres = self._calculate_regular_grid_coordsinates(y = y_lr, x = x_lr,
                                                                num_y = num_y,
                                                                num_x = num_x)

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

    def convert_micrometer_to_pixel(self,
                                    zyx: tuple[float],
                                    pyramid_level: Optional[int] = None) -> tuple[int]:
        """
        Convert micrometers to pixels for a given pyramid level.

        Parameters:
            zyx (tuple): The micrometer coordinates in the form (z, y, x) to be converted
                to pixels.
            pyramid_level (int): The pyramid level (resolution), to which the output
                pixel coordinates will refer to.  If `None`, the lowest-resolution
                (highest) pyramid level will be selected.
        
        Returns:
            A tuple (z, y, x) with pixel coordinates.
        
        Examples:
            Obtain the x-length (x_px) of a scale bar corresopnding to 10 micrometer for
            pyramid level 0:

            >>> z_px, y_px, x_px = plateA.convert_micrometer_to_pixel(zyx=(0,0,10), pyramid_level=0)
        """
        # digest arguments
        assert isinstance(zyx, tuple)
        pyramid_level = self._digest_pyramid_level_argument(pyramid_level)

        # convert
        pyramid_spacing = self.level_zyx_spacing[pyramid_level]
        zyx_px = np.round(np.divide(zyx, pyramid_spacing)).astype(int)

        # return as tuple
        return(tuple(zyx_px))

    def convert_pixel_to_micrometer(self,
                                    zyx: tuple[int],
                                    pyramid_level: Optional[int] = None) -> tuple[float]:
        """
        Convert pixels to micrometers for a given pyramid level.

        Parameters:
            zyx (tuple): The pixel coordinates in the form (z, y, x) to be converted
                to micrometers.
            pyramid_level (int): The pyramid level (resolution), to which the input
                pixel coordinates refer to. If `None`, the lowest-resolution
                (highest) pyramid level will be selected.
        
        Returns:
            A tuple (z, y, x) with micrometer coordinates.
        
        Examples:
            Obtain the micrometer dimensions of a cube with 10 pixel sides for pyramid level 0:

            >>> z_um, y_um, x_um = plateA.convert_pixel_to_micrometer(zyx=(10,10,10), pyramid_level=0)
        """
        # digest arguments
        pyramid_level = self._digest_pyramid_level_argument(pyramid_level)

        # convert
        pyramid_spacing = self.level_zyx_spacing[pyramid_level]
        zyx_um = np.round(np.multiply(zyx, pyramid_spacing)).astype(float)

        # return as tuple
        return(tuple(zyx_um))

    def convert_pixel_to_pixel(self,
                               zyx: tuple[int],
                               pyramid_level_from: Optional[int] = None,
                               pyramid_level_to: Optional[int] = None) -> tuple[int]:
        """
        Convert pixel coordinates between pyramid levels.

        Parameters:
            zyx (tuple): The pixel coordinates in the form (z, y, x) to be converted.
            pyramid_level_from (int): The pyramid level (resolution), to which the input
                pixel coordinates refer to. If `None`, the lowest-resolution
                (highest) pyramid level will be selected.
            pyramid_level_to (int): The pyramid level (resolution), to which the output
                pixel coordinates will refer to. If `None`, the lowest-resolution
                (highest) pyramid level will be selected.
        
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
        zyx_scale = self.level_zyx_scalefactor**(pyramid_level_from - pyramid_level_to)
        zyx_new = np.round(np.multiply(zyx, zyx_scale)).astype(int)

        # return as tuple
        return(tuple(zyx_new))


    # analysis methods --------------------------------------------------------
    def calc_average_FOV(self,
                         include_wells: list[str] = [],
                         pyramid_level: Optional[int] = None,
                         channel: int = 0) -> np.ndarray:
        """
        Calculate the average field of view.
         
        Using the coordinates stored in table 'FOV_ROI_table', calculate the averge
        field of view for wells in `include_wells`, for `channel` at resolution `pyramid_level`.

        Parameters:
            include_wells (list): List of well names to include. If empty `[]`, all wells are included.
            pyramid_level (int): The pyramid level (resolution level), from which the image
                should be extracted. If `None`, the lowest-resolution (highest) pyramid level
                will be selected.
            channel (int): The channel for which the fields of view should be averaged.

        Returns:
            The averaged field of view, as an array of shape (z,y,x).

        Examples:
            Calculate the averaged field of view for channel zero over all wells
            for pyramid level 1.

            >>> avg_fov = plateA.calc_average_FOV(pyramid_level = 1, channel = 0)
        """
        # check if required data is available
        if not 'FOV_ROI_table' in self.table_names:
            raise ValueError("`FOV_ROI_table` not found - cannot calculate average FOV")
        
        # digest arguments
        include_wells = self._digest_include_wells_argument(include_wells)
        pyramid_level = self._digest_pyramid_level_argument(pyramid_level)

        # extract FOV table and scaling information
        fov_tab = self.get_table('FOV_ROI_table')
        pyramid_spacing = self.level_zyx_spacing[pyramid_level]
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







# FractalZarrSet class ------------------------------------------------------
class FractalZarrSet:
    """Represents a folder containing one or several ome-zarr fileset(s)."""

    # constructor and helper functions ----------------------------------------
    def __init__(self, path: str, name = None) -> None:
        """
        Initializes a container for a folder containing one or several ome-zarr
        fileset(s) (.zarr). Typically, the object is used for a folder which
        contains exactly two related .zarr objects, one corresponding to the
        four-dimensional (c,z,y,x) plate dataset, and a second one corresponding to
        a three-dimensional maximum intensity projection derived from it.

        Parameters:
            path (str): Path containing the ome-zarr fileset(s).
            name (str): Optional name for the experiment.
        
        Examples:
            Get an object corresponding to a set of .zarr's.

            >>> from ez_zarr import hcs_wrappers
            >>> plate_set = hcs_wrappers.FractalZarrSet('path/to/zarrs')
            >>> plate_set

            This will print information on the zarrs.
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
            raise ValueError(f'no .zarr filesets found in `{path}`')
        self.zarr: list[FractalZarr] = [FractalZarr(os.path.join(self.path, f)) for f in self.zarr_paths]
        self.zarr_names: list[str] = [x.name for x in self.zarr]
        self.zarr_mip_idx: Optional[int] = None
        self.zarr_3d_idx: Optional[int] = None
        if len(self.zarr) == 2 and self.zarr_names[0].replace('_mip.zarr', '.zarr') == self.zarr_names[1]:
            # special case of 3D plate plus derived maximum intensity projection?
            self.zarr_mip_idx = 0
            self.zarr_3d_idx = 1

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
