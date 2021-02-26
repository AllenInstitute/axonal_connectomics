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


def deskew_and_save_tiff(cfg):

	inputfile = cfg["raw_file"]
	outputfile = cfg["deskewed_file"] 
	xypixelsize = cfg['xypixelsize']
	angle = cfg ['angle']
	
	dz = np.sin(angle*np.pi/180.0)
	dx = xypixelsize

	deskewfactor = np.cos(angle*np.pi/180.0)/dx
	dzdx_aspect = dz/dx

	print("Parameter summary:")
	print("==================")
	print("dx, dy:", dx)
	print("dz:", dz)
	print("deskewfactor:", deskewfactor)
	print("voxel aspect ratio z-voxel/xy-voxel:", dzdx_aspect)


	vol = io.get_tiff_image(inputfile)
	# deskew 
	print("Now deskewing...")
	starttime = time.time()
	skew = np.eye(4)
	skew[2,0] = deskewfactor
	output_shape = get_output_dimensions(skew, vol)
	deskewed = affine_transform(vol, np.linalg.inv(skew), output_shape=output_shape,order=1)
	endtime = time.time()
	print("Done deskewing!, %d seconds"%(endtime-starttime))
	io.save_tiff_image(deskewed,outputfile)
