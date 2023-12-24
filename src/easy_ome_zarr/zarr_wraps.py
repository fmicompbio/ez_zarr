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

    def _load_channel_info(self, well = None):
        """Load info about available channels."""
        if not well: # pick first well
            well = self.wells[0]['path']
        well_group = self.top[os.path.join(well, '0')] # convention: single field of view per well
        if not 'omero' in well_group.attrs or not 'channels' in well_group.attrs['omero']:
            raise ValueError(f"no channel info found in well {well}")
        return well_group.attrs['omero']['channels']
    
    def _load_multiscale_info(self, well = None):
        """Load info about available scales."""
        if not well: # pick first well
            well = self.wells[0]['path']
        well_group = self.top[os.path.join(well, '0')] # convention: single field of view per well
        if not 'multiscales' in well_group.attrs:
            raise ValueError(f"no multiscale info found in well {well}")
        return well_group.attrs['multiscales'][0]
    
    def _load_label_names(self, well = None):
        """Load label names (available segmentations)."""
        if not well: # pick first well
            well = self.wells[0]['path']
        label_path = os.path.join(well, '0', 'labels')
        if label_path in self.top:
            return self.top[label_path].attrs['labels']
        else:
            return []

    def _load_table_names(self, well = None):
        """Load table names (can be extracted using .get_table())."""
        if not well: # pick first well
            well = self.wells[0]['path']
        table_path = os.path.join(well, '0', 'tables')
        if table_path in self.top:
            return self.top[table_path].attrs['tables']
        else:
            return []
    
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
    
    def get_path(self):
        """Gets the path of the ome-zarr fileset."""
        return self.path

    def get_wells(self):
        """Gets info on wells in the ome-zarr fileset."""
        return self.wells

    def get_channels(self):
        """Gets info on channels in the ome-zarr fileset."""
        return self.channels

    def get_table(self, table_name, include_wells = None, as_AnnData = False):
        """Extract table for wells in a ome-zarr fileset."""
        if not include_wells: # include all wells
            include_wells = [x['path'] for x in self.wells]

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




# FractalFmiZarr class ------------------------------------------------------
class FractalFmiZarr:
    """Represents a folder containing one or several ome-zarr fileset(s)."""

    def __init__(self, folder_path):
        self.path = folder_path
