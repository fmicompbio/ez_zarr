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
            
            # make sure that we have at least one array
            if len(self.array_dict) == 0:
                raise ValueError(f"{self.path} does not contain any arrays")
            
        # load info about available scales in image and labels
        if not 'multiscales' in self.zarr_group.attrs:
            raise ValueError(f"{self.path} does not contain a 'multiscales' attribute")
        self.multiscales_image = self._load_multiscale_info(self.zarr_group, skip_checks)
        self.multiscales_labels = {x: self._load_multiscale_info(self.zarr_group.labels[x], skip_checks) for x in self.label_names}

        # load channel metadata
        self.channels = []
        if 'omero' in self.zarr_group.attrs and 'channels' in self.zarr_group.attrs['omero']:
            self.channels = self.zarr_group.attrs['omero']['channels']
       
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
    def _extract_zyx_spacings(dataset_dict: dict[str, Any]) -> list[str, list[float]]:
        if 'path' not in dataset_dict or 'coordinateTransformations' not in dataset_dict or 'scale' not in dataset_dict['coordinateTransformations'][0]:
            raise ValueError("could not extract zyx spacing from multiscale_info")
        return [dataset_dict['path'], dataset_dict['coordinateTransformations'][0]['scale']]
    
    # string representation ---------------------------------------------------
    def __str__(self):
        nch = len(self.channels)
        chlabs = ', '.join([x['label'] for x in self.channels])
        npl = len(self.multiscales_image['datasets'])
        segnames = ', '.join(self.label_names)
        tabnames = ', '.join(self.table_names)
        image_zyx = {x[0]: x[1][1:] for x in [self._extract_zyx_spacings(y) for y in self.multiscales_image['datasets']]}
        pl_scalefactor = 'None (only one pyramid level)'
        if len(image_zyx) > 1:
            pl_nms = list(image_zyx.keys())
            pl_scalefactor = np.divide(image_zyx[pl_nms[1]], image_zyx[pl_nms[0]])
        return f"Image {self.name}\n  path: {self.path}\n  n_channels: {nch} ({chlabs})\n  n_pyramid_levels: {npl}\n  pyramid_zyx_scalefactor: {pl_scalefactor}\n  full_resolution_zyx_spacing (µm): {image_zyx[list(image_zyx.keys())[0]]}\n  segmentations: {segnames}\n  tables (measurements): {tabnames}\n"
    
    def __repr__(self):
        return str(self)
    
