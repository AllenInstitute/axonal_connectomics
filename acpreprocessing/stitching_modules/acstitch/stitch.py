# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 13:39:58 2023

@author: kevint
"""
import numpy
from acpreprocessing.stitching_modules.acstitch.sift_stitch import SiftDetector,generate_rois_from_pointmatches
from acpreprocessing.stitching_modules.acstitch.ccorr_stitch import get_correspondences
from acpreprocessing.stitching_modules.acstitch.zarrutils import get_group_from_src
from acpreprocessing.stitching_modules.acstitch.io import read_pointmatch_file


def generate_sift_pointmatches(p_srclist,q_srclist,miplvl=0,sift_kwargs=None,stitch_kwargs=None):
    p_datasets = [get_group_from_src(src)[miplvl] for src in p_srclist]
    q_datasets = [get_group_from_src(src)[miplvl] for src in q_srclist]
    sd = SiftDetector(**sift_kwargs)
    if "sift_pointmatch_file" in stitch_kwargs and stitch_kwargs["sift_pointmatch_file"]:
        sift_pmlist = read_pointmatch_file(stitch_kwargs["sift_pointmatch_file"])
    else:
        sift_pmlist = None
    if sift_pmlist is None:
        p_ptlist,q_ptlist = sd.stitch_over_segments(p_datasets,q_datasets,**stitch_kwargs) # zstarts, zlength, stitch_axes, i_slice, j_slice, ny, dy
    else:
        roilist = generate_rois_from_pointmatches(pm_list=sift_pmlist,**stitch_kwargs) # axis_range, roi_dims, stitch_axes, ij_shift, nx, dx
        p_ptlist,q_ptlist = sd.stitch_over_rois(p_datasets,q_datasets,roilist,**stitch_kwargs)
    pmlist = []
    if not p_ptlist is None:
        for p_src,q_src,p_pts,q_pts in zip(p_srclist,q_srclist,p_ptlist,q_ptlist):
            if not p_pts is None and len(p_pts) > 0:
                pmlist.append({"p_tile":p_src,"q_tile":q_src,"p_pts":p_pts,"q_pts":q_pts})
            else:
                pmlist.append({"p_tile":p_src,"q_tile":q_src,"p_pts":None,"q_pts":None})
    return pmlist


def generate_ccorr_pointmatches(p_srclist,q_srclist,miplvl=0,ccorr_kwargs=None,stitch_kwargs=None):
    if "sift_pointmatch_file" in stitch_kwargs and stitch_kwargs["sift_pointmatch_file"]:
        print("running crosscorrelation with points from " + stitch_kwargs["sift_pointmatch_file"])
        sift_pmlist = read_pointmatch_file(stitch_kwargs["sift_pointmatch_file"])
    else:
        sift_pmlist = None
    p_datasets = [get_group_from_src(src)[miplvl] for src in p_srclist]
    q_datasets = [get_group_from_src(src)[miplvl] for src in q_srclist]
    pmlist = []
    for i in range(len(p_datasets)):
        print("computing pointmatches for source pair " + str(i))
        pds = p_datasets[i]
        qds = q_datasets[i]
        if not sift_pmlist is None:
            if i < len(sift_pmlist) and not sift_pmlist[i]["p_pts"] is None and len(sift_pmlist[i]["p_pts"])>0:
                ppts,qpts = run_ccorr_with_sift_points(pds, qds, sift_pmlist[i]["p_pts"].astype(int), sift_pmlist[i]["q_pts"].astype(int), **ccorr_kwargs)
            else:
                ppts = None
                qpts = None
        else:
            ppts,qpts = run_ccorr(**ccorr_kwargs)
        if not ppts is None and len(ppts) > 0:
            pmlist.append({"p_tile":p_srclist[i],"q_tile":q_srclist[i],"p_pts":ppts,"q_pts":qpts})
        else:
            pmlist.append({"p_tile":p_srclist[i],"q_tile":q_srclist[i],"p_pts":None,"q_pts":None})
    return pmlist
    

def run_ccorr_with_sift_points(p_ds,q_ds,p_siftpts,q_siftpts,n_cc_pts=1,axis_w=[32,32,32],pad_array=False,axis_shift=[0,0,0],axis_range=None,cc_threshold=0.8,**kwargs):
    p_pts,q_pts = get_cc_points_from_sift(p_ds, q_ds, p_siftpts, q_siftpts,n_cc_pts,axis_shift,axis_range)
    ppm,qpm = get_correspondences(p_ds,q_ds,p_pts,q_pts,numpy.asarray(axis_w),pad_array,cc_threshold=cc_threshold)
    return ppm,qpm

def get_cc_points_from_sift(p_ds,q_ds,p_siftpts,q_siftpts,n_cc_pts=1,axis_shift=[0,0,0],axis_range=None):
    # TODO: handle overly granular bins with potentially 0 sift points returned
    if axis_range is None:
        axis_range = [[] for i in range(p_siftpts.shape[1])]
    if len(axis_range[0]) == 0:
        zstarts = numpy.linspace(numpy.min(p_siftpts[:,0]),numpy.max(p_siftpts[:,0]),n_cc_pts+1)
    p_pts = numpy.empty((n_cc_pts,3),dtype=int)
    q_pts = numpy.empty((n_cc_pts,3),dtype=int)
    for i in range(n_cc_pts):
        r = numpy.full(p_siftpts.shape[0],True)
        for ai,a in enumerate(axis_range):
            if len(a)>0:
                r = r & ((p_siftpts[:,ai]>=a[0]) & (p_siftpts[:,ai]<=a[1]))
            elif ai == 0:
                r = r & ((p_siftpts[:,ai]>=zstarts[i]) & (p_siftpts[:,ai]<=zstarts[i+1]))
        pr = p_siftpts[r]
        if len(pr) > 0:
            imax = numpy.argmax(p_ds[0,0,pr[:,0],pr[:,1],pr[:,2]])
            ppt = pr[imax,:]
        else:
            ppt = numpy.array([(zstarts[i]+zstarts[i+1])/2,numpy.mean(p_siftpts[:,1]),numpy.mean(p_siftpts[:,2])],dtype=int)
        p_pts[i] = ppt
        q_pts[i] = ppt + numpy.array(axis_shift)
    return p_pts,q_pts
    
    
    
def run_ccorr(**kwargs):
    pass