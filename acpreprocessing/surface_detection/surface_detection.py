import numpy as np
from skimage.filters import threshold_otsu
from scipy.ndimage import maximum_filter, median_filter,convolve, binary_dilation
from scipy.interpolate import griddata,RBFInterpolator
from acpreprocessing.stitching_modules.acstitch.zarrutils import get_group_from_src

def my_threshold(data):
    thresh = threshold_otsu(data)
    #thresh = np.percentile(data,95)
    return thresh

def preconvolve(data,size):
    convolved = np.empty(data.shape,dtype=data.dtype)
    k = np.ones((size,size))
    for i in range(data.shape[0]):
        convolved[i,...] = convolve(data[i,...],k,mode="constant",cval=0)
    return convolved

def predilation(data,radius):
    d = int(2*radius + 1)
    k = np.zeros((d,d,d),dtype=int)
    r = radius
    for x in range(d):
        for y in range(d):
            for z in range(d):
                if (x-r)**2 + (y-r)**2 + (z-r)**2 <= r**2:
                    k[x,y,z] = 1
    return binary_dilation(data,structure=k).astype(int)

def premedian(data,size):
    medianed = np.empty(data.shape,dtype=data.dtype)
    for i in range(data.shape[0]):
        medianed[i,...] = median_filter(data[i,...],size=(size,size))
    return medianed

def premax(data,size):
    M = np.empty(data.shape,dtype=data.dtype)
    for i in range(data.shape[0]):
        M[i,...] = maximum_filter(data[i,...],size=(size,size))
    return M

def get_first_z(maskstack,flip=False):
    M = maskstack.astype(int)
    if flip:
        M = np.flip(maskstack,axis=0)
    dims = M.shape # (Nz,Nx,Ny)
    zs = np.zeros((dims[1],dims[2]),dtype='int')
    
    for i1 in range(dims[1]):
        for i2 in range(dims[2]):
            z = np.nonzero(M[:,i1,i2])
            if len(z[0])>0:
                if not flip:
                    zs[i1,i2] = z[0][0]
                else:
                    if z[0][0] > 0:
                        zs[i1,i2] = dims[0] - z[0][0] - 1
                    else:
                        zs[i1,i2] = 0
            else:
                zs[i1,i2] = 0
            
    return zs

def make_surface_map_tps(zIn,gridsize,miplvl,surfsup=False):
    dims = zIn.shape
    d0 = int(np.floor(dims[0]/gridsize[0]))
    d1 = int(np.floor(dims[1]/gridsize[1]))
    surfmap = np.zeros(gridsize,dtype=int)
    mapy = np.zeros(gridsize,dtype=int)
    mapx = np.zeros(gridsize,dtype=int)
    for i0 in range(gridsize[0]):
        for i1 in range(gridsize[1]):
            zIni = zIn[d0*i0:d0*(i0+1),d1*i1:d1*(i1+1)]
            if surfsup:
                z = np.max(zIni[zIni>0]) if np.any(zIni>0) else 0
                y,x = np.unravel_index(np.argmax(zIni, axis=None), zIni.shape)
            else:
                z = np.min(zIni[zIni>0]) if np.any(zIni>0) else 0
                y,x = np.unravel_index(np.argmin(zIni, axis=None), zIni.shape)
            surfmap[i0,i1] = z*(2**miplvl)
            mapy[i0,i1] = int((d0*i0 + y)*(2**miplvl))
            mapx[i0,i1] = int((d1*i1 + x)*(2**miplvl))
    return surfmap,mapy,mapx


#surfsup = False for S32, True for S33
def detect_surface(zarr_path,cutout,miplvl=2,surfsup=False):
    mipdata = get_group_from_src(srcpath=zarr_path)[miplvl]
    zstart = cutout["z"][0]
    zlength = cutout["z"][1] - zstart
    ylength = cutout["y"][1] - cutout["y"][0]
    z0 = int(zstart/(2**miplvl))
    z1 = z0 + int(zlength/(2**miplvl))
    A = mipdata[0,0,z0:z1,:,:]
    thresh = my_threshold(A)
    print(thresh)
    B = A > thresh
    C = premax(premedian(B.astype(int),size=5),size=10)
    D = predilation(C,radius=4)
    E = get_first_z(D.transpose((2,1,0)),flip=surfsup)
    Z,Y,X = make_surface_map_tps(E,gridsize=(10,20),miplvl=miplvl,surfsup=surfsup)
    tpsy,tpsx = np.meshgrid(np.arange(ylength,step=2**miplvl,dtype=int),np.arange(zlength,step=2**miplvl,dtype=int),indexing='ij')
    tpsyx = np.hstack((tpsy.flatten()[:,np.newaxis],tpsx.flatten()[:,np.newaxis]))
    YX = np.concatenate((Y[Z>0].flatten()[:,np.newaxis],X[Z>0].flatten()[:,np.newaxis]),axis=1)
    F = RBFInterpolator(YX,Z[Z>0].flatten(),smoothing=1,kernel='thin_plate_spline')(tpsyx)
    gridy,gridx = np.meshgrid(np.arange(ylength,dtype=int),np.arange(zlength,dtype=int),indexing='ij')
    G = griddata((tpsy.flatten(),tpsx.flatten()),F,(gridy,gridx),method='nearest')
    return G