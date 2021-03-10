# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 19:21:02 2021

@authors: kevint, clayr, sharmishtaas, russelt
"""

import math
import numpy as np
import skimage.measure
from skimage.transform import downscale_local_mean as skids


def area_average_downsample(array, block_shape, dtype=None):
    """Block downsample ndarray by mean pooling.  Use only when array
        dimensions are evenly divisible by block dimensions.

    Parameters
    ----------
    array : numpy.ndarray
        n-dimensional numpy array
    block_shape : tuple
        shape of blocks for pooling, effectively downsample
        factor for each axis
    dtype : data-type, optional
        numpy-compatible dtype of output array.  Defaults to dtype of array

    Returns
    -------
    ds_array : numpy.ndarray
        mean-pooled downsampling of array
    """
    output_shape = tuple(
        int(math.ceil(s / f)) for s, f in zip(array.shape, block_shape))
    temp = np.zeros(output_shape, float)
    counts = np.zeros(output_shape, int)
    for offset in np.ndindex(block_shape):
        part = array[tuple(np.s_[o::f]
                     for o, f in zip(offset, block_shape))]
        indexing_expr = tuple(np.s_[:s] for s in part.shape)
        temp[indexing_expr] += part
        counts[indexing_expr] += 1
    return np.cast[dtype or array.dtype](temp / counts)


def downsample_stack_legacy(imstack, dsfactor=4):
    """area downsample a 3d zxy volume on the xy axes.

    Parameters
    ----------
    imstack : numpy.ndarray
        3D numpy array representing which will be downsampled on axes 1 and 2
    dsfactor : int
        dowsample factor on downsampled axes

    Returns
    -------
    ds_imstack : numpy.ndarray
        3D numpy array with axes 1 and 2 downsampled by dsfactor
    """
    dims = imstack.shape  # (Nz,Nx,Ny)
    dims_ds = (int(dims[0]), int(dims[1]/dsfactor), int(dims[2]/dsfactor))

    dsstack = np.zeros(dims_ds, dtype=float)
    for l in range(dims[0]):
        dsstack[l, :, :] = skids(imstack[l, :, :].astype(float),
                                 (dsfactor, dsfactor))

    return dsstack


def downsample_stack(imstack, dsfactor=4, method=None, dtype=float):
    """area downsample a 3D zxy volume on the xy axes.  This is a
        convenience function with a switch for different methods
        producing the same results (with performance differences)

    Parameters
    ----------
    imstack : numpy.ndarray
        3D numpy array representing which will be downsampled on axes 1 and 2
    dsfactor : int
        dowsample factor on downsampled axes
    method : string, optional
        method from "area_average_downsample", "block_reduce", "legacy". If
        None will default to "block_reduce" or "area_average_downsample"
        based on shape of imstack
    dtype : data-type, optional
        numpy compatible dtype for output array.  Defaults to float

    Returns
    -------
    ds_imstack : numpy.ndarray
        3D numpy array with axes 1 and 2 downsampled by dsfactor
    """
    # dtype is float by default to match legacy, but can be changed
    if dtype is None:
        dtype = imstack.dtype

    default_method = (
        "block_reduce"
        if any([d % dsfactor for d in imstack.shape[1:]])
        else "area_average_downsample")
    ds_functions = {
        "area_average_downsample": lambda x: area_average_downsample(
            x, (1, dsfactor, dsfactor), dtype),
        "block_reduce": lambda x: skimage.measure.block_reduce(
            x, (1, dsfactor, dsfactor),
            func=np.mean).astype(dtype),
        "legacy": lambda x: downsample_stack_legacy(x, dsfactor)
        }
    dsstack = ds_functions.get(
        method, ds_functions[default_method])(
            imstack)
    return dsstack


def clip_and_adjust_range_values(stack,clippercentile=95,minforpercentile=500,maxval=255):
    p = np.percentile(stack[stack>minforpercentile],clippercentile)
    stack = stack*maxval/p
    stack[stack<0] = 0
    stack[stack>maxval] = maxval
    return stack
