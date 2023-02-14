import os
from threading import Thread
import numpy as np
from datetime import datetime
from time import sleep

class PixelShiftDeskew(object):

#implements (chunked) pixel shifting deskew
#class parameters:
#skew_dims_zyx = dimensions of skewed (input) tiff data (xy are camera coordinates, z is tiff chunk #size, xz define skewed plane and y is non-skewed axis)
#stride = number of camera (x) pixels to shift onto a sample (z') plane (sample z dim = camera x #dim/stride)
#deskewFlip = flip volume (reflection, parity inversion)
#dtype = datatype of input data

#NOTE: must be run sequentially as each tiff chunk contains data for the next deskewed block #retained in self.slice1d except for the final chunk which should form the rhomboid edge

    def __init__(self,skew_dims_zyx,stride=1,deskewFlip=False,dtype='uint16',**kwargs):
        sdims = skew_dims_zyx
        blockdims = (int(sdims[2]/stride),sdims[1],stride*sdims[0])
        subblocks = int(np.ceil((sdims[2]+stride*sdims[0])/(stride*sdims[0])))
        print(subblocks)
        blockx = sdims[0]
        dsi = []
        si = []
        for i_block in range(subblocks):
            sxv = []
            szv = []
            for sz in range(blockx):
                sxstart = i_block*stride*blockx-stride*sz
                sxend = (i_block+1)*stride*blockx-stride*sz
                if sxstart < 0:
                    sxstart = 0
                if sxend > sdims[2]:
                    sxend = sdims[2]
                sx = np.arange(sxstart,sxend)
                sxv.append(sx)
                szv.append(sz*np.ones(sx.shape,dtype=sx.dtype))
            sxv = np.concatenate(sxv)
            szv = np.concatenate(szv)
            dsx = sxv + stride*szv - i_block*stride*blockx
            dsz = np.floor(sxv/stride).astype(int)
            dsi.append(np.ravel_multi_index((dsz,dsx),(blockdims[0],blockdims[2])))
            si.append(np.ravel_multi_index((szv,sxv),(sdims[0],sdims[2])))
        self.dsi = dsi
        self.si = si
        self.slice1d = np.zeros((subblocks,blockdims[1],blockdims[2]*blockdims[0]),dtype=dtype)
        self.blockdims = blockdims
        self.subblocks = subblocks
        self.flip = deskewFlip
        self.dtype = dtype
        
    def deskew_block(self,blockData,n):
        subb = self.subblocks
        block3d = np.zeros(self.blockdims,dtype=self.dtype)
        zdim = block3d.shape[0]
        ydim = block3d.shape[1]
        xdim = block3d.shape[2]
        print('deskewing block ' + str(n))
        print(blockData.shape)
        order = (np.arange(subb)+n)%subb
        for y in range(ydim):
            for i,o in enumerate(order):
                # flip stack axis 2 for ispim2
                s = -1 if self.flip else 1
                self.slice1d[o,y,:][self.dsi[i]] = blockData[:,y,::s].ravel()[self.si[i]]
            block3d[:,y,:] = self.slice1d[n%subb,y,:].reshape((zdim,xdim))
            self.slice1d[n%subb,y,:]=0
        return block3d
    
    def reshape_joined_shapes(self,joined_shapes):
        return