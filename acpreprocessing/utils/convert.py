# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 19:21:02 2021

@author: kevint, sharmishtaas
"""

import os
import numpy as np
import tifffile as tf
from skimage.transform import downscale_local_mean as skids
from PIL import Image

import re
import time



def downsample_stack(imstack,dsfactor=4):
    
    dims = imstack.shape # (Nz,Nx,Ny)
    dims_ds = (int(dims[0]),int(dims[1]/dsfactor),int(dims[2]/dsfactor))
    
    dsstack = np.zeros(dims_ds,dtype=float)
    for l in range(dims[0]):
        dsstack[l,:,:] = skids(imstack[l,:,:].astype(float),(dsfactor,dsfactor))
        
        
    return dsstack

def clip_and_adjust_range_values(stack,clippercentile=95,minforpercentile=500,maxval=255):
    p = np.percentile(stack[stack>minforpercentile],clippercentile)
    stack = stack*maxval/p
    stack[stack<0] = 0
    stack[stack>maxval] = maxval
    return stack

