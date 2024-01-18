"""Wrap ome-zarr filesets in a class.

Represent an ome-zarr fileset as a class to give high-level
access to its contents.
"""

__all__ = ['FmiZarr', 'FractalFmiZarr']
__version__ = '0.1'
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


# FmiZarr class ---------------------------------------------------------------
class FmiZarr:
    """Represents a ome-zarr fileset."""

    # constructor and helper functions ----------------------------------------
    def __init__(self, zarr_path, name = None):
        """
        Initializes an ome-zarr fileset (.zarr) from its path.
        Typically, the fileset represents a single assay plate, and 
        we assume that the structures (pyramid levels, labels, table, etc.)
        are consistent across wells.

        Parameters:
            zarr_path (str): Path containing the plate ome-zarr fileset.
            name (str): Optional name for the plate.
        
        Examples:
            Get an object corresponding to a plate.

            >>> from easy_ome_zarr import zarr_wraps
            >>> plateA = zarr_wraps.FmiZarr('path/to/plate.zarr')
            >>> plateA

            This will print information on the plate.
        """

        self.path = zarr_path
        if name:
            self.name = name
        else:
            self.name = os.path.basename(self.path)
        self.top = zarr.open_group(store = self.path, mode = 'r')
        if not 'plate' in self.top.attrs:
            raise ValueError(f"{self.name} does not contain a zarr fileset with a 'plate'")
        self.acquisitions = self.top.attrs['plate']['acquisitions']
        self.columns = self.top.attrs['plate']['columns']
        self.rows = self.top.attrs['plate']['rows']
        self.wells = self.top.attrs['plate']['wells']
        self.channels = self._load_channel_info()
        self.multiscales = self._load_multiscale_info()
        self.level_paths = [x['path'] for x in self.multiscales['datasets']]
        self.level_zyx_spacing = [x['coordinateTransformations'][0]['scale'][1:] for x in self.multiscales['datasets']] # convention: unit is micrometer
        self.level_zyx_scalefactor = np.divide(self.level_zyx_spacing[1], self.level_zyx_spacing[0])
        self.label_names = self._load_label_names()
        self.table_names = self._load_table_names()

    def _load_channel_info(self):
        """Load info about available channels."""
        well = self.wells[0]['path']
        well_group = self.top[os.path.join(well, '0')] # convention: single field of view per well
        if not 'omero' in well_group.attrs or not 'channels' in well_group.attrs['omero']:
            raise ValueError(f"no channel info found in well {well}")
        return well_group.attrs['omero']['channels']
    
    def _load_multiscale_info(self):
        """Load info about available scales."""
        well = self.wells[0]['path']
        well_group = self.top[os.path.join(well, '0')] # convention: single field of view per well
        if not 'multiscales' in well_group.attrs:
            raise ValueError(f"no multiscale info found in well {well}")
        return well_group.attrs['multiscales'][0]
    
    def _load_label_names(self):
        """Load label names (available segmentations)."""
        well = self.wells[0]['path']
        label_path = os.path.join(well, '0', 'labels')
        if label_path in self.top:
            return self.top[label_path].attrs['labels']
        else:
            return []

    def _load_table_names(self):
        """Load table names (can be extracted using .get_table())."""
        well = self.wells[0]['path']
        table_path = os.path.join(well, '0', 'tables')
        if table_path in self.top:
            return self.top[table_path].attrs['tables']
        else:
            return []
    
    # utility functions -------------------------------------------------------
    def _digest_well_argument(self, well = None):
        """Interpret a single `well` argument in the context of a given FmiZarr object."""
        if not well:
            # no well given -> pick first one
            return self.wells[0]['path']
        else:
            return os.path.join(well[:1].upper(), well[1:])

    def _digest_include_wells_argument(self, include_wells = None):
        """Interpret an `include_wells` argument in the context of a given FmiZarr object."""
        if not include_wells: 
            # no wells given -> include all wells
            include_wells = [x['path'] for x in self.wells]
        else:
            # transform well names from 'B03' format to path format 'B/03'
            include_wells = [self._digest_well_argument(w) for w in include_wells]
        return include_wells

    def _digest_pyramid_level_argument(self, pyramid_level = None):
        """Interpret a `pyramid_level` argument in the context of a given FmiZarr object."""
        if pyramid_level == None: 
            # no pyramid level given -> pick lowest resolution one
            pyramid_level = int(self.level_paths[-1])
        else:
            # make sure it is an integer
            pyramid_level = int(pyramid_level)
        return pyramid_level
    
    def _calculate_regular_grid_coordsinates(self, y, x, num_y = 10, num_x = 10):
        """
        Calculate the cell coordinates for a regular rectangular grid of total size (y, x)
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
        return f"FmiZarr {self.name}\n  path: {self.path}\n  n_wells: {nwells}\n  n_channels: {nch} ({chlabs})\n  n_pyramid_levels: {npl}\n  pyramid_zyx_scalefactor: {self.level_zyx_scalefactor}\n  full_resolution_zyx_spacing: {self.level_zyx_spacing[0]}\n  segmentations: {segnames}\n  tables (measurements): {tabnames}\n"
    
    def __repr__(self):
        return str(self)
    
    # accessors ---------------------------------------------------------------
    def get_path(self):
        """Gets the path of the ome-zarr fileset."""
        return self.path

    def get_wells(self, simplify = False):
        """Gets info on wells in the ome-zarr fileset."""
        if simplify:
            return [w['path'].replace('/', '') for w in self.wells]
        else:
            return self.wells

    def get_channels(self):
        """Gets info on channels in the ome-zarr fileset."""
        return self.channels
    
    def get_table_names(self):
        """Gets list of table names in the ome-zarr fileset."""
        return self.table_names

    # query methods -----------------------------------------------------------
    def get_table(self, table_name, include_wells = None, as_AnnData = False):
        """Extract table for wells in a ome-zarr fileset."""
        include_wells = self._digest_include_wells_argument(include_wells)

        table_paths = [os.path.join(w, '0', 'tables', table_name) for w in include_wells]
        # remark: warn if not all well have the table?
        table_paths = [p for p in table_paths if p in self.top]

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

    def get_image_rect(self, well = None, pyramid_level = None,
                       upper_left = None, lower_right = None, width_height = None,
                       as_NumPy = False):
        """
        Extract a rectangular image region (all z planes if several) from a well by coordinates.

        None or at least two of `upper_left`, `lower_right` and `width_height` need to be given.
        If none are given, it will return the full image.
        Otherwise, `upper_left` contains the lower indices than `lower_right`
        (origin on the top-left, zero-based coordinates), and each of them is a tuple of (x, y).
        """
        # digest arguments
        well = self._digest_well_argument(well)
        pyramid_level = self._digest_pyramid_level_argument(pyramid_level)

        # load image (convention: single field of view per well -> '0')
        img_path = os.path.join(self.path, well, '0', str(pyramid_level))
        img = dask.array.from_zarr(img_path)

        # calculate corner coordinates and subset if needed
        # (images are always of 4D shape c,z,y,x)
        num_unknowns = sum([x == None for x in [upper_left, lower_right, width_height]])
        if num_unknowns == 1:
            if width_height:
                assert all([x > 0 for x in width_height])
                if not upper_left:
                    upper_left = tuple(lower_right[i] - width_height[i] for i in range(2))
                elif not lower_right:
                    lower_right = tuple(upper_left[i] + width_height[i] for i in range(2))
            assert all([upper_left[i] < lower_right[i] for i in range(len(upper_left))])
            img = img[:,
                      :,
                      slice(upper_left[1], lower_right[1] + 1),
                      slice(upper_left[0], lower_right[0] + 1)]
        elif num_unknowns != 3:
            raise ValueError("Either none or two of `upper_left`, `lower_rigth` and `width_height` have to be given")

        # convert if needed and return
        if as_NumPy:
            img = np.array(img)
        return img

    def get_image_sampled_rects(self, well = None,
                                pyramid_level = None, lowres_level = None,
                                num_x = 10, num_y = 10,
                                num_select = 9,
                                sample_method = 'sum',
                                channel = 0,
                                seed = 42,
                                as_NumPy = False):
        """
        Split a well image into a regular grid and extract a subset of grid cells (all z planes if several).

        `num_x` and `num_y` define the grid by specifying the number of cell in x and y.
        `num_select` picks that many from the total number of grid cells and returns them as a list. All returned grid cells are guaranteed to be of equal size, but a few pixels in the
        last row or column may not be included if the image shape is not divisible by `num_x` or `num_y`.

        `sample_method` defines how the `num_select` cells are selected. Possible values are:
          - 'sum': order grid cells decreasingly by the sum of `channel` (working on `lowres_level`)
          - 'var': order grid cells decreasingly by the variance of `channel` (working on `lowres_level`)
          - 'random': order grid cells randomly (use `seed`)
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
    
    # analysis methods --------------------------------------------------------
    def calc_average_FOV(self, include_wells = None, pyramid_level = None, channel = 0):
        """
        Calculate the average field of view for wells in `include_wells`,
        at resolution `pyramid_level`, for `channel`.
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
        avg_fov = None
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
                if avg_fov is None:
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







# FractalFmiZarr class ------------------------------------------------------
class FractalFmiZarr:
    """Represents a folder containing one or several ome-zarr fileset(s)."""

    # constructor and helper functions ----------------------------------------
    def __init__(self, path, name = None):
        if not os.path.isdir(path):
            raise ValueError(f'`{path}` does not exist')
        self.path = path
        if name is None:
            self.name = os.path.basename(self.path)
        else:
            self.name = name
        self.zarr_paths = [f for f in os.listdir(self.path) if f[-5:] == '.zarr']
        if len(self.zarr_paths) == 0:
            raise ValueError(f'no .zarr filesets found in `{path}`')
        self.zarr = [FmiZarr(os.path.join(self.path, f)) for f in self.zarr_paths]
        self.zarr_names = [x.name for x in self.zarr]
        self.zarr_mip_idx = None
        self.zarr_3d_idx = None
        if len(self.zarr) == 2 and self.zarr_names[0].replace('_mip.zarr', '.zarr') == self.zarr_names[1]:
            # special case of 3D plate plus derived maximum intensity projection?
            self.zarr_mip_idx = 0
            self.zarr_3d_idx = 1

    # string representation ---------------------------------------------------
    def __str__(self):
        nplates = len(self.zarr)
        platenames = ''.join(f'    {i}: {self.zarr[i].name}\n' for i in range(len(self.zarr)))
        return f"FractalFmiZarr {self.name}\n  path: {self.path}\n  n_plates: {nplates}\n{platenames}\n"
    
    def __repr__(self):
        return str(self)

    # accessors ---------------------------------------------------------------
    def __len__(self):
        return len(self.zarr)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.zarr[key]
        elif isinstance(key, str):
            return self.zarr[self.zarr_names.index(key)]