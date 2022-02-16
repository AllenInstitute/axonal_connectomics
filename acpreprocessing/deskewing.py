import numpy as np
import tifffile
import matplotlib.pyplot as plt
from numpy.linalg import inv
import json
import time
from scipy.ndimage import affine_transform
import logging
logging.getLogger("tifffile").setLevel(logging.ERROR)
from acpreprocessing.utils import io
from acpreprocessing.utils import convert
import numpy as np
import os
from PIL import Image
import time


def transformed_vol_dims(aff_mtx, shape):
    """get output dimensions of a 3D volume given a 4x4 affine transform.
      Rounds up to match results in lls_dd.

    Parameters
    ----------
    aff_mtx: numpy.ndarray
        4x4 affine transformation matrix
    shape: iterable
        shape of 3d volume to be transformed

    Returns
    -------
    dims:
        expected dimensions of output volume
    """
    d0, d1, d2 = [s-1 for s in shape]
    homogeneous_corner_pts = np.array([
        [0, 0, 0, 1],
        [d0, 0, 0, 1],
        [0, d1, 0, 1],
        [0, 0, d2, 1],
        [d0, d1, 0, 1],
        [d0, 0, d2, 1],
        [0, d1, d2, 1],
        [d0, d1, d2, 1]
    ])
    # nonhomogeneous transformed corner points
    tformed_corner_pts = np.dot(aff_mtx, homogeneous_corner_pts.T).T[:, :-1]
    # add 1 to peak-to-peak to avoid fencepost
    dims = tformed_corner_pts.ptp(axis=0) + 1
    # round to next multiple of 2
    dims = (np.ceil(dims / 2.) * 2).astype(int)
    return dims


def deskew_from_config(vol,config):

	xypixelsize = config['xypixelsize']
	angle = config ['angle']
	dzstage = config['zstage']

	dz = np.sin(angle*np.pi/180.0)*dzstage
	dx = xypixelsize
	deskewfactor = np.cos(angle*np.pi/180.0)*dzstage/dx
	dzdx_aspect = dz/dx

	print("deskewing..")
	skew = np.eye(4)
	skew[2,0] = deskewfactor
	print(skew)
	output_shape = transformed_vol_dims(skew, vol.shape)
	deskewed = affine_transform(vol, np.linalg.inv(skew),output_shape=output_shape,order=1)
	return deskewed
