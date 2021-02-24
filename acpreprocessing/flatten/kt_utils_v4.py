# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 11:36:30 2020

@author: kevint
"""

import numpy as np
import scipy as sp
from skimage.filters.rank import maximum
from skimage.morphology import disk
from skimage.filters import gaussian
from skimage.io import imsave
from skimage.transform import resize

def applyIJFiltersStack(imstack,max_radius=64,smooth_radius=128):
    dims = imstack.shape # (Nz,Nx,Ny)

    IJFstack = np.zeros(dims)
    for l in range(dims[0]):
        frame = imstack[l,:,:]
        filtered = applyIJFilters(frame,max_radius,smooth_radius)
        IJFstack[l,:,:] = filtered
    return IJFstack

def makeIJMask(filtered,threshold):
    return filtered > threshold

def applyIJFilters(frame,max_radius,smooth_radius):
    frame_max = maximum(frame,disk(max_radius))
    frame_smooth = gaussian(frame_max,sigma=smooth_radius,preserve_range=True)
    return frame_smooth

def saveMaskStack(savefile,maskstack):
    tiffstack = maskstack.astype('uint8')*255
    imsave(savefile,tiffstack)
    print('Mask stack saved to ' + savefile)
    
def getFirstLastIndices(maskstack):
    dims = maskstack.shape # (Nz,Nx,Ny)
    Zstart = np.zeros((dims[1],dims[2]),dtype='int')
    Zend = np.zeros((dims[1],dims[2]),dtype='int')
    
    diffstack = np.concatenate((maskstack[0,:,:][np.newaxis,:,:],np.diff(maskstack,axis=0)),axis=0)
    
    for i1 in range(dims[1]):
        for i2 in range(dims[2]):
            zs = np.nan
            ze = np.nan
            inds = np.argwhere(diffstack[:,i1,i2])
            if len(inds) == 0:
                # No first slice found : error condition
                print('ERROR: No first slice found!!!')
            elif len(inds) == 1:
                # Only first slice found : last slice is end of stack
                zs = inds[0][0]
                ze = dims[0]-1
            elif len(inds) == 2:
                # First and last slice found
                zs = inds[0][0]
                ze = inds[1][0]
            else:
                # More than two True values found : error condition, use last index
                print('ERROR: More than 2 transitions found!!!')
                zs = inds[0][0]
                ze = inds[-1][0]
            Zstart[i1,i2] = zs
            Zend[i1,i2] = ze
            
    return Zstart,Zend

def fixcurvedstack(imstack,Zstart,Zend):
    dims = imstack.shape
    newstack = np.zeros(dims,dtype=imstack.dtype)
    for l1 in range(dims[1]):
        for l2 in range(dims[2]):
            zs = Zstart[l1,l2]
            ze = Zend[l1,l2]
            if zs is not None:
                newstack[0:int(ze-zs),l1,l2] = imstack[int(zs):int(ze),l1,l2]
            else:
                print('ERROR: no start slice found for pixel (' + str(l1) + ',' + str(l2) + ')')
                
    return newstack

def fixcurvedstack_interp(imstack,Zstart,Zend):
    #CR 20200226: new version to interpolate data points so that the output stack has a constant thickness
    # New variable Zthickness
    Zthickness = Zend - Zstart
    maxthickness = Zthickness.max()
    dims = imstack.shape
    newstack = np.zeros([maxthickness,dims[1],dims[2]],dtype=imstack.dtype)
    for i1 in range(dims[1]):
        for i2 in range(dims[2]):
            zs = Zstart[i1,i2]
            ze = Zend[i1,i2]
            if zs is not None:
                interp_zseries = np.interp(np.arange(maxthickness)*1.*(ze-zs)/maxthickness, 
                                              np.arange(ze-zs), imstack[int(zs):int(ze),i1,i2])
                newstack[:,i1,i2] = interp_zseries
            else:
                print('ERROR: no start slice found for pixel (' + str(i1) + ',' + str(i2) + ')')
                
    return newstack

def upsampleIndices(inds_ds,upfactor=4):
    '''
    upsampleIndices uses skimage.transform.resize for convenience
    probably better to write custom function for controlling behavior and edge cases
    verify output sensibility until then
    '''
    if np.any(np.isnan(inds_ds),axis=None):
        print('WARNING: slice indices contains NaN values - expect unexpected behavior')
    else:
        print('upsampling by ' + str(upfactor))
    dims = inds_ds.shape
    
    inds_up = resize(inds_ds.astype(float),(upfactor*dims[0],upfactor*dims[1])).astype(int)
    if np.any(inds_up<0):
        print('ERROR: interpolated indices go out of bounds')
        inds_up[inds_up<0] = 0
        
    return inds_up

def minMaxThreshold(fstack):
    minMaxImage = np.max(fstack,axis=0)
    print('Threshold maximum estimate: ' + str(np.min(minMaxImage,axis=None)))
    return minMaxImage

def medMinThreshold(fstack):
    minImage = np.min(fstack,axis=0)
    print('Threshold minimum estimate: ' + str(np.median(minImage,axis=None)))
    return minImage
    
