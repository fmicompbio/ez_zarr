"""Wrap ome-zarr filesets in a class.

Represent an ome-zarr fileset as a class to give high-level
access to its contents.
"""

__all__ = ['FmiZarr', 'FractalFmiZarr']
__version__ = '0.1'
__author__ = 'Silvia Barbiero, Michael Stadler'


# imports -------------------------------------------------------------------
import os, re
import numpy as np
import zarr
import dask
import anndata as ad
# import tifffile
import warnings


# FmiZarr class -------------------------------------------------------------
class FmiZarr:
    """Represents a ome-zarr fileset."""

    # constructor and helper functions
    def __init__(self, zarr_path, name = None):
        """
        Initializes an ome-zarr fileset (.zarr) from its path.
        We assume that the structure (pyramid levels, labels, table, etc.)
        are consistent across wells.
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
    
    # utility functions
    def _digest_well_argument(self, well = None):
        """Interpret a single `well` argument in the context of a given FmiZarr object."""
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
        if not pyramid_level: 
            # no pyramid level given -> pick lowest resolution one
            pyramid_level = self.level_paths[-1]
        else:
            # make sure it is an integer
            pyramid_level = int(pyramid_level)
        return pyramid_level

    # string representation
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
    
    # slot accessors
    def get_path(self):
        """Gets the path of the ome-zarr fileset."""
        return self.path

    def get_wells(self):
        """Gets info on wells in the ome-zarr fileset."""
        return self.wells

    def get_channels(self):
        """Gets info on channels in the ome-zarr fileset."""
        return self.channels

    # query methods
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
            return anndata_combined.to_df()

    def get_image_rect(self, well = None, pyramid_level = None,
                       upper_left = None, lower_right = None, width_height = None,
                       as_NumPy = False):
        """
        Extract a rectangular image region (all z planes if several) from a well by coordinates.
        None or at least two of `upper_left`, `lower_right` and `width_height` need to be given.
        If none are given, it will return the full image.
        Otherwise, `upper_left` contains the lower indices than `lower_right` (origin on the top-left).
        Each of them is a tuple of (x, y).
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
            img = img[:, :, range(upper_left[1], lower_right[1] + 1), range(upper_left[0], lower_right[0] + 1)]
        elif num_unknowns != 3:
            raise ValueError("Either none are two of `upper_left`, `lower_rigth` and `width_height` have to be given")

        # convert if needed and return
        if as_NumPy:
            img = np.array(img)
        return img






# FractalFmiZarr class ------------------------------------------------------
class FractalFmiZarr:
    """Represents a folder containing one or several ome-zarr fileset(s)."""

    def __init__(self, folder_path):
        self.path = folder_path
