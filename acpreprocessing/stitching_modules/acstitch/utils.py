import numpy as np
import os
import zarr
import cv2 as cv
import json
import gzip

def get_lists(basePath,baseName,acqName,segNum,centerTile,xIndList,yIndList,xShift=[],yShift=[]):
    fpaths = []
    shifts = []
    tiles = []
    for i,x in enumerate(xIndList):
        for j,y in enumerate(yIndList):
            if not (x==centerTile[0] and y==centerTile[1]):
                tstr = str(x) + '_' + str(y) + '_' + str(segNum)
                runDir = baseName + '_' + acqName + '_' + str(x)
                stripDir = acqName + '_' + str(x) + '_Pos' + str(y)
                tifName = acqName + '_' + tstr + '.tif'
                fpaths.append(os.path.join(basePath,runDir,stripDir,tifName))
                tiles.append(tstr)
                if not xShift:
                    m = int(centerTile[0]-x)
                else:
                    m = xShift[i]
                if not yShift:
                    n = int(centerTile[1]-y)
                else:
                    n = yShift[j]
                shifts.append([m,n])
    return fpaths,tiles,shifts

def get_dataset(zpath,grpname,miplvl=0):
    zf = zarr.open(zpath)
    return zf[grpname][miplvl]

def get_cc_windows(imgRef,img=None,pixShifts=[],support=[0.1,0.1]):
    clahe = cv.createCLAHE(clipLimit=5.0,tileGridSize=(8,8))
    if img is None:
        img = imgRef
    dshape = [int(np.around(d*support[i])) for i,d in enumerate(imgRef.shape)]
    start = [int(np.floor((d-dshape[i])/2)) for i,d in enumerate(imgRef.shape)]
    ccRef = imgRef[start[0]:start[0]+dshape[0],start[1]:start[1]+dshape[1]]
    ccRef = clahe.apply(ccRef)
    if not len(pixShifts)==2:
        pixShifts = [0,0]
    ss = [0,0]
    indShifts = pixShifts[::-1]
    for i in range(2):
        if start[i]+indShifts[i]-dshape[i]<0:
            ss[i] = 0
        elif start[i]+indShifts[i]+2*dshape[i]>= img.shape[i]:
            ss[i] = img.shape[i]-3*dshape[i]
        else:
            ss[i] = start[i] + indShifts[i] - dshape[i]
    ccTarget = img[ss[0]:ss[0]+3*dshape[0],ss[1]:ss[1]+3*dshape[1]]
    ccTarget = clahe.apply(ccTarget)
    return ccRef,ccTarget

def get_pixel_shift_xy(ij,resultShape):
    cij = (np.array(resultShape)-1)/2
    xy_shift = (np.array(ij) - cij)[::-1].astype(int)
    return xy_shift

