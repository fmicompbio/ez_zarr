"""Helpful functions for plotting and working with image arrays."""

__all__ = ['zproject']
__version__ = '0.1.3'
__author__ = 'Silvia Barbiero, Michael Stadler'


# imports ---------------------------------------------------------------------
import dask.array
import numpy as np
from typing import Union, Optional
# global variables ------------------------------------------------------------
plate_layouts = {
    '6well': {'rows': ['A','B'],
              'columns': [str(i+1).zfill(2) for i in range(3)]},
    '24well': {'rows': ['A','B','C','D'],
               'columns': [str(i+1).zfill(2) for i in range(6)]},
    '96well': {'rows': ['A','B','C','D','E','F','G','H'],
               'columns': [str(i+1).zfill(2) for i in range(12)]},
    '384well': {'rows': ['A','B','C','D','E','F','G','H',
                         'I','J','K','L','M','N','O','P'],
                'columns': [str(i+1).zfill(2) for i in range(24)]},
}

             axis: Optional[int]=1,
             method: Optional[str]='maximum',
             keepdims: Optional[bool]=True,
             img_bit: Optional[int]=16) -> Union[dask.array.Array, np.ndarray]:
    """
    Project a 4D or 3D image along z-axis according to desired projection method.

    Helper function to conveniently project multi-dimensional images along z-axis according to desired projection method. 

    Parameters:
        data (dask.array or np.ndarray): The array to be projected
        axis (int): The axis on which to perform the projection (typically
            the z-axis). Unless otherwise specified, it defaults to axis 1 (by convention Fractal outputs have shape ch,z,y,x). Note: if masks are projected, Fractal output have usually shape z,y,x, so use `axis=0` in that case.
        method (str): The projection method of choice. Available are 
            'maximum' (default), 'minimum' (typically used for brightfield images), 'sum', 'average'
        keepdims (bool): Should the output array keep the same dimensions 
            as the input array, or should the z-axis been squeezed (default behaviour of e.g. np.min())
        img_bit (int): Is the image 8- or 16-bit? Relevant for 'sum' 
            projection, where clipping might be necessary
    
    Returns:
        The input array where the z-axis has been projected according to the method of choice.
    
    Examples:
        Project np.array along z-axis:

        >>> zproject(data=np.array([[[1,2,4],[1,3,5],[1,6,7]],[[1,2,4],[1,3,5],[1,6,7]]]), method='maximum', axis=1)
    """
    methods = {
        'maximum': np.max,
        'minimum': np.min,
        'sum': np.sum,
        'average': np.mean,
    }

    if not method in methods:
        raise ValueError(f"Unknown method ({method}), should be one of: " + ', '.join(list(methods.keys())))
    
    im = methods[method](data, axis=axis, keepdims=keepdims) 
    if method == 'sum':
        max_value = (2**img_bit) - 1
        im = np.clip(im, 0, max_value)

    return im
