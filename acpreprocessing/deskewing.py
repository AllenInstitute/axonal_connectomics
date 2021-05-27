import numpy as np
import tifffile
import matplotlib.pyplot as plt
from numpy.linalg import inv
import json
import time
from lls_dd.transform_helpers import *
from lls_dd.transforms import *
from scipy.ndimage import affine_transform
import logging
logging.getLogger("tifffile").setLevel(logging.ERROR)
from .utils import io
from .utils import convert
import numpy as np
import os
from PIL import Image
import time

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
	output_shape = get_output_dimensions(skew, vol)
	deskewed = affine_transform(vol, np.linalg.inv(skew),output_shape=output_shape,order=1)
	return deskewed