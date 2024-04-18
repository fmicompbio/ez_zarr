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
            
            # make sure that we have at least one array
            if len(self.array_dict) == 0:
                raise ValueError(f"{self.path} does not contain any arrays")
            
        # load info about available scales in image and labels
        if not 'multiscales' in self.zarr_group.attrs:
            raise ValueError(f"{self.path} does not contain a 'multiscales' attribute")
        self.multiscales_image = self._load_multiscale_info(self.zarr_group, skip_checks)
        self.multiscales_labels = {x: self._load_multiscale_info(self.zarr_group.labels[x], skip_checks) for x in self.label_names}

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
        return f"Image {self.name}\n  path: {self.path}\n  n_channels: {nch} ({chlabs})\n  n_pyramid_levels: {npl}\n  pyramid_{zyx}_scalefactor: {pl_scalefactor}\n  full_resolution_{zyx}_spacing: {image_spacings[list(image_spacings.keys())[0]]}\n  segmentations: {segnames}\n  tables (measurements): {tabnames}\n"
    
    def __repr__(self):
        return str(self)
    
