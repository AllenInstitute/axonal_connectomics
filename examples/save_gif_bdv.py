# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 19:21:02 2021

@author: kevint
"""

import os
import numpy as np
import tifffile as tf
from skimage.transform import downscale_local_mean as skids
from PIL import Image

from multiprocessing import Process, Pool
from functools import partial

# CR removed this package and commented out the bdv h5 stuff
#import npy2bdv as nb
import re
import time

#def run(filepath,savepath,savename):
def run(savename, mntpath, subpath, savepath):
    subpath_file = subpath + savename
    savepath_file = savepath + savename
    filepath = os.path.join(mntpath, subpath_file)
    
    TIFFSIZE = 510
    DSSHAPE = (512,512)
    filelist = getfilelist(filepath)
    dsDataStack = np.zeros((len(filelist)*TIFFSIZE,DSSHAPE[0],DSSHAPE[1]),dtype='uint16')
    savedir = os.path.abspath(savepath_file)
    setup_paths(savedir)
    iftimer = True
    writeindex = 0
    
    for tiff in filelist:
        try:
            print('Processing ' + os.path.basename(tiff))
            time0=time.time()
            newname = os.path.basename(tiff) + '_ds.gif'
            fpath = os.path.join(savedir,'GIFs',newname)
            
            data = tf.imread(tiff,multifile=False)
            if iftimer: print ("Read file: ", time.time() - time0)  
            dsdata = downsample_stack(data)
            if iftimer: print ("downsampled ", time.time() - time0) 
            ds8 = make8bit(dsdata)
            if iftimer: print ("converted to 8 bit ", time.time() - time0)  
            imgs = [Image.fromarray(img) for img in ds8]
            imgs[0].save(fpath, save_all=True, append_images=imgs[1:], duration=25, loop=0)
            if iftimer: print ("wrote gif ", time.time() - time0)
            
            fi = writeindex
            li = writeindex+data.shape[0]
            dsDataStack[fi:li,...] = dsdata
            writeindex = li
        except:
            pass
    #CR removed bigdataviewer h5 stuff   
    """
    fpath = os.path.join(savedir,'BDV',savename)

    shear_x_px = 1
    affine_matrix = np.array(((1.0, 0.0, -shear_x_px, 0.0),
                              (0.0, 1.0, 0.0, 0.0),
                              (0.0, 0.0, 1.0, 0.0)))
    bdv_writer = nb.BdvWriter(fpath + '_bdv.h5')
    bdv_writer.append_view(dsDataStack, m_affine=affine_matrix, name_affine="shearing transformation",)
    bdv_writer.write_xml_file()
    bdv_writer.close()
    """
    print('Done processing files in ' + filepath)

def getfilelist(dirpath):
    fpath = os.path.abspath(dirpath)
    if os.path.exists(fpath):
        flist = [os.path.join(fpath,f) for f in os.listdir(dirpath) if os.path.isfile(os.path.join(dirpath, f)) and f.endswith('.tif')]
        flist.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
        print(flist)
    else:
        print('ERROR: ' + fpath + ' does not exist')
        flist = []
    
    return flist

def setup_paths(savedir):
    if not os.path.exists(savedir):
        os.makedirs(savedir)
        print(savedir + ' created')
    gifpath = os.path.join(savedir,'GIFs')
    if not os.path.exists(gifpath):
        os.makedirs(gifpath)
        print(gifpath + ' created')
    bdvpath = os.path.join(savedir,'BDV')
    if not os.path.exists(bdvpath):
        os.makedirs(bdvpath)
        print(bdvpath + ' created')
    
def downsample_stack(imstack,dsfactor=4):
    
    dims = imstack.shape # (Nz,Nx,Ny)
    dims_ds = (int(dims[0]),int(dims[1]/dsfactor),int(dims[2]/dsfactor))
    
    dsstack = np.zeros(dims_ds,dtype=float)
    for l in range(dims[0]):
        dsstack[l,:,:] = skids(imstack[l,:,:].astype(float),(dsfactor,dsfactor))
        
    #CR print('Downsampled by ' + str(dsfactor))
        
    return dsstack

def make8bit(stack,clippercentile=95):
    #CR new variable clippercentile gives the percentile that is set to 255, was 90 in KT version
    #CR new varialbe minforpercentile gives the minimum value of the images that go into the percentile calculation, was 500
    minforpercentile = 500
    p = np.percentile(stack[stack>minforpercentile],clippercentile)
    stack = stack*255/p
    stack[stack<0] = 0
    stack[stack>255] = 255
    
    return stack

if __name__=='__main__':
    savepositions = ['14', '15']
    for i in range(len(savepositions)):
        savepositions[i] = 'Pos0' + savepositions[i]
    subpath = "487748_49_NeuN_NFH_488_25X_0.5XPBS/global_l40_"
    savepath = "/synology/Axon_backup/Dropbox/Dropbox/Pooja/487748_49_NeuN_NFH_488_25X_0.5XPBS/ds_"
    mntpath = "/ispim1_data/"

    pool = Pool(processes=5)
    run_savename = partial(run, mntpath=mntpath, subpath=subpath, savepath=savepath)
    pool.map(run_savename, savepositions)

    '''
    for pos in savepositions:
        #savename = 'Pos0' + pos
        savename = pos
        subpath = "PoojaB/487748_37_NeuN_NFH_488_25X_1XPBS/lightsheet_l50_"
        savepath = "/synology/Axon_backup/Dropbox/Dropbox/Pooja/487748_37_NeuN_NFH_488_25X_1XPBS/ds_"

        #run(os.path.join(mntpath,subpath),savepath,savename)
        run(savename, mntpath, subpath, savepath)
    '''