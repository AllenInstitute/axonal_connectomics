# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 13:39:58 2023

@author: kevint
"""

from sift_stitch import SiftDetector
from zarrutils import get_group_from_src

def generate_sift_pointmatches(p_srclist,q_srclist,miplvl=0,sift_kwargs=None,stitch_kwargs=None):
    p_datasets = [get_group_from_src(src)[miplvl] for src in p_srclist]
    q_datasets = [get_group_from_src(src)[miplvl] for src in q_srclist]
    sd = SiftDetector(**sift_kwargs)
    p_ptlist,q_ptlist = sd.stitch_over_segments("zx",p_datasets,q_datasets,**stitch_kwargs) # zstarts, zlength, i_slice, j_slice, ny, dy)
    pmlist = []
    for p_src,q_src,p_pts,q_pts in zip(p_srclist,q_srclist,p_ptlist,q_ptlist):
        pmlist.append({"p_tile":p_src,"q_tile":q_src,"p_pts":p_pts,"q_pts":q_pts})
    return pmlist