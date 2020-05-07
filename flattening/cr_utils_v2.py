# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 11:36:30 2020

@author: clay reid
"""

# CR: 2020-0422 
# plotker, plotkers are old routines for plotting spatial receptive fields, but will also be useful for
# plotting image stacks. For now, let's use "ker" to stand in for a single image, kers for an image stack
import numpy as np
#import math
import matplotlib.pyplot as plt
IFDBG = False # for verbose mode etc.
IFZEROCENTERED = False # for verbose mode etc.

def plotker(ker,axes=None, title='',cmap='bwr'):  # or 'gray'
  if axes is None:
    axes = plt.gca()
  axes.axis('off')
  if ker.dtype == complex:
    kertmp= compleximg2rgb(ker)
    axes.imshow(kertmp,interpolation="none")
  else:
    # for a regular (on and off) real-valued kernel
    if ker.ndim == 2:
        scl = np.max(np.abs(ker))
        print("in plotker, scl = " + str(scl))
        if (IFZEROCENTERED):
            axes.imshow(ker,interpolation="none", cmap=cmap, vmax=scl, vmin=-1.*scl)
        else:
            axes.imshow(ker,interpolation="none", cmap=cmap, vmax=scl, vmin=0)        
    axes.set_title(title)
    #plt.show()

#plot multiple kers using subplots
# ker2 is optional. If it exists, it should have the same dimensions of ker and will
# be shown on alternating rows for comparison
def plotkers(ker, ker2 = None, nkers=0, dims = None,ifpanels=False,axes=None,title='',ifnorm=True,cmap='gray'): 
    print(IFDBG)
    nkerdims = ker.shape
    if nkers == 0:  # number of kernels to show
        nkers = nkerdims[0]
    if IFDBG:
        print(nkerx)
    nkerx = nkerdims[1]
    nkery=nkerdims[2]
    print(ker.dtype)
    if dims is None:
        nrows = np.int(np.floor(np.sqrt(nkers)))
        ncols = np.int(np.floor(nkers/nrows))
    else:
        nrows = int(dims[0])
        ncols = int(dims[1])
        #had to be typecast, please confirm no loss of precision
    if (ifpanels):    # old approach, not recommended because wastes space
        # AND IT DOESN'T work if there is a second kernel, ker2, or for complex
        f, axarr = plt.subplots(nrows,ncols)
        for i in range(nrows):
            for j in range(ncols):
                ind = i*ncols+j
                kertmp = ker[ind, :,:].copy()
                scl = np.max(np.abs(kertmp))
                f = axarr[i,j].imshow(kertmp,cmap=cmap, vmax=scl, vmin=-1.*scl,interpolation="none")
                kerplotformat(f)
        plt.tight_layout()
        return f,axarr
    else:
        # kertmp has an extra row and column for each kernel that is
        #set to 1, to make a border, the original kernel is scaled to -1, 1

        nkertmp = nrows*ncols
        # border between kernels should depend on kernel sz:
        nborder = np.int(np.max([np.ceil(nkerx/20),1]))
        # To delete border:    nborder = 0   # for color it is too difficult.... because ydim is rgbrgb...
        kersml=ker[0:nkertmp,0:nkerx,0:nkery].copy()
        '''
        # Defining the global scale factor before the scltmp normalization (ifnorm = True) results in over-normalization
        # Moved to line 94 where I believe it behaves as expected
        scl = np.max(np.abs(kersml))  # global scale factor
        print(str(scl))
        '''
        if ifnorm: # normalize all kernels
            for i in range(nkertmp):
                scltmp=max(np.max(np.abs(kersml[i,:,:])),0.0001) # to avoid zero divide
                #print(i, scltmp)
                if scltmp > 0.0001:
                    kersml[i,:,:] = kersml[i,:,:]/scltmp
                else:
                    kersml[i,:,:] = 0.0001
        kersml = kersml.reshape([nrows,ncols,nkerx,nkery])
        kersml = kersml.transpose(0,2,1,3)
        
        scl = np.max(np.abs(kersml))  # global scale factor
        print(str(scl)) # should be 1.0 now if ifnorm = True
        
        if ker2 is None:     # only one kernel
            kertmp = np.ones([nrows,nkerx+nborder,ncols,nkery+nborder])
            if ker.dtype == complex:  # for orientations
                kertmp = np.complex(1)*kertmp
            kertmp[:,0:nkerx,:,0:nkery] = kersml/scl
        else:       # alternate rows of the two kernels
            scl2 = max(np.max(np.abs(ker2[0:nkerx,0:nkery,0:nkertmp])),0.0001)
            ker2sml=ker2[0:nkertmp,0:nkerx,0:nkery].copy()
            ker2sml = ker2sml.reshape([nrows,ncols,nkerx,nkery])
            ker2sml = ker2sml.transpose(0,2,1,3)
            nrows = 2 * nrows
            kertmp = np.ones([nrows,nkerx+nborder,ncols,nkery+nborder])
            kertmp[0:nrows:2,0:nkerx,:,0:nkery] = kersml/scl
            kertmp[1:nrows:2,0:nkerx,:,0:nkery] = ker2sml/scl2
            # old kertmp = np.ones([nrows,nkerx+nborder,ncols,nkery+nborder])
            if ker.dtype == complex:
                kertmp = np.complex(1)*kertmp
        # kertmp = kertmp.reshape([nkerx+nborder,nkery+nborder,nrows,ncols])
        # transpose dims to make kernel dimensions fastest and 3rd fastest
        kertmp = kertmp.reshape([(nkerx+nborder)*nrows,(nkery+nborder)*ncols])
        plotker(kertmp,axes=axes,title=title,cmap=cmap)

