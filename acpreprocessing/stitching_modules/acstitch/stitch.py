# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 13:39:58 2023

@author: kevint
"""
import numpy
import argschema
from acpreprocessing.stitching_modules.acstitch.sift_stitch import generate_rois_from_pointmatches,stitch_over_rois,stitch_over_segments
from acpreprocessing.stitching_modules.acstitch.ccorr_stitch import get_correspondences
from acpreprocessing.stitching_modules.acstitch.zarrutils import get_zarr_array
from acpreprocessing.stitching_modules.acstitch.io import read_pointmatch_file,save_pointmatch_file


def run_ccorr(p_ds,q_ds,p_pts,q_pts,n_cc_pts=1,axis_w=[32,32,32],r=1,pad_array=False,axis_shift=[0,0,0],axis_range=None):
    ppm,qpm = get_correspondences(p_ds,q_ds,p_pts,q_pts,numpy.asarray(axis_w),r=r,pad=pad_array)
    return ppm,qpm

def run_sift(p_ds,q_ds,miplvl=0,sift_kwargs=None,stitch_kwargs=None):
    if "sift_pointmatch_file" in stitch_kwargs and stitch_kwargs["sift_pointmatch_file"]:
        sift_pmlist = read_pointmatch_file(stitch_kwargs["sift_pointmatch_file"])
    else:
        sift_pmlist = None
    if sift_pmlist is None:
        if "roi_list" in stitch_kwargs and not stitch_kwargs["roi_list"] is None:
            p_ptlist,q_ptlist = stitch_over_rois(sift_kwargs,p_ds,q_ds,**stitch_kwargs)
        else:
            p_ptlist,q_ptlist = stitch_over_segments(sift_kwargs,p_ds,q_ds,**stitch_kwargs)
    else:
        roilist = generate_rois_from_pointmatches(pm_list=sift_pmlist,**stitch_kwargs)
        p_ptlist,q_ptlist = stitch_over_rois(sift_kwargs,p_ds,q_ds,roilist,**stitch_kwargs)
    return p_ptlist,q_ptlist

def get_dataset_from_path(tilepath,miplvl=0):
    return get_zarr_array(tilepath,miplvl=miplvl)

def run_stitch_method(p_tilepath,q_tilepath,p_points,q_points,stitch_method,miplvl=0,sift_kwargs=None,stitch_kwargs=None):
    stitch_kwargs = ({} if stitch_kwargs is None else stitch_kwargs)
    if stitch_method == "ccorr":
        kwargs = {"p_ds":get_dataset_from_path(p_tilepath,miplvl),
                  "q_ds":get_dataset_from_path(q_tilepath,miplvl),
                  "p_pts": p_points/(2**miplvl),
                  "q_pts": q_points/(2**miplvl)}

        ccorr_kwargs = {
            "axis_w": stitch_kwargs.get("axis_w", [32, 32, 32]),
            "r": stitch_kwargs.get("r", 1),
            "pad_array": stitch_kwargs.get("pad_array", False),
        }
        p_pts,q_pts = run_ccorr(**kwargs,**ccorr_kwargs)
    elif stitch_method == "sift":
        kwargs = {"p_ds":get_dataset_from_path(p_tilepath,miplvl),
                  "q_ds":get_dataset_from_path(q_tilepath,miplvl),
                  "miplvl":miplvl}
        p_pts,q_pts = run_sift(**kwargs,sift_kwargs=sift_kwargs,stitch_kwargs=stitch_kwargs)
    else:
        return run_stitch_method(p_tilepath,q_tilepath,p_points,q_points,stitch_method="ccorr",stitch_kwargs=stitch_kwargs)
    return {"p_tile":p_tilepath,"q_tile":q_tilepath,"p_pts":p_pts*(2**miplvl),"q_pts":q_pts*(2**miplvl)}


def stitch_tiles_from_pmfile(input_file,output_file,stitch_method,miplvl=0,sift_kwargs=None,stitch_kwargs=None):    
    in_pms = read_pointmatch_file(input_file)
    print(in_pms)
    out_pms = []
    for tspec in in_pms:
        args = [tspec.get(key) for key in ["p_tile","q_tile","p_pts","q_pts"]]
        pmdict = run_stitch_method(*args,stitch_method=stitch_method,miplvl=miplvl,sift_kwargs=sift_kwargs,stitch_kwargs=stitch_kwargs)
        # else:
        #     print("WARNING: tspec contains None")
        #     pmdict = None
        out_pms.append(pmdict)
    save_pointmatch_file(out_pms,output_file)
    

class StitchTilesParameters(argschema.ArgSchema):
    input_file = argschema.fields.Str(required=True)
    output_file = argschema.fields.Str(required=True)
    stitch_method = argschema.fields.Str(required=True)
    miplvl = argschema.fields.Int(required=False,default=0)
    sift_kwargs = argschema.fields.Dict(required=False, default=None, allow_none=True)
    stitch_kwargs = argschema.fields.Dict(required=False, default=None, allow_none=True)


class StitchTiles(argschema.ArgSchemaParser):
    default_schema = StitchTilesParameters
    def run(self):
        # CHANGED: forward all optional args, not just the first three.
        stitch_tiles_from_pmfile(
            self.args['input_file'],
            self.args['output_file'],
            self.args['stitch_method'],
            self.args['miplvl'],
            self.args['sift_kwargs'],
            self.args['stitch_kwargs'],
        )


if __name__ == "__main__":
    mod = StitchTiles()
    mod.run()