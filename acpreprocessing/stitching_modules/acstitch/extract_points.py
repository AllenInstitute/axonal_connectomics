# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 11:19:46 2025

@author: kevint
"""

import argschema
import numpy
import json
from skimage.feature import blob_log,blob_dog,blob_doh
from acpreprocessing.stitching_modules.acstitch.zarrutils import get_zarr_group,get_zarr_array,ZarrV3Metadata
from acpreprocessing.stitching_modules.acstitch.io import save_pointmatch_file

DEFAULT_METHOD = blob_dog

def detect_blobs(data,method=None,**kwargs):
    # input 3d image data array
    # return blobs detected by method
    if method == "log":
        blob_func = blob_log
    elif method == "dog":
        blob_func = blob_dog
    elif method == "doh":
        blob_func = blob_doh
    else:
        blob_func = DEFAULT_METHOD
        
    return blob_func(data,**kwargs)


def detect_blobs_roi(zarray,z_range=None,y_range=None,x_range=None,method=None,blob_kwargs=None):
    # input zarr array and roi defined by ranges (at miplvl)
    # return blobs detected in roi
    def get_roi(zarray,z_range,y_range,x_range):
        return zarray[0,0,z_range[0]:z_range[1],y_range[0]:y_range[1],x_range[0]:x_range[1]]
    blob_kwargs = ({} if blob_kwargs is None else blob_kwargs)
    
    data = get_roi(zarray,z_range,y_range,x_range)
    blobs = detect_blobs(data,method,**blob_kwargs)
    if len(blobs)>0:
        values = data[blobs[:,0].astype(int),blobs[:,1].astype(int),blobs[:,2].astype(int)]
        blobs[:,:3] = blobs[:,:3] + numpy.array([[z_range[0],y_range[0],x_range[0]]])
        return blobs,values
    print("no blobs detected!!")
    return None,None


def extract_points(zarray,roi_list,method,blob_kwargs=None):
    # input tile path to run method with blob kwargs at mip level
    # return
    blobs = []
    values = []
    for roi in roi_list:
        blobs_roi,values_roi = detect_blobs_roi(zarray,roi["z"],roi["y"],roi["x"],method=method,blob_kwargs=blob_kwargs)
        if not blobs_roi is None:
            blobs.append(blobs_roi)
            values.append(values_roi)
    return blobs,values
    

def save_extracted_pointmatch_file(src_tile,trg_tile,output_file,miplvl,roi_list,method,blob_kwargs=None,n_points=1,offset=None):
    if offset is None:
        p_md = ZarrV3Metadata(zgroup=get_zarr_group(src_tile))
        q_md = ZarrV3Metadata(zgroup=get_zarr_group(trg_tile))
        offset = numpy.divide(q_md.get_coordinate_translation(),q_md.mip_voxel_dims(miplvl))-numpy.divide(p_md.get_coordinate_translation(),p_md.mip_voxel_dims(miplvl))
    zarray = get_zarr_array(src_tile,miplvl=miplvl)
    src_points_rois,values_rois = extract_points(zarray,roi_list,method,blob_kwargs)
    src_points = []
    for points,values in zip(src_points_rois,values_rois):
        if len(points) > 0:
            if len(points) < n_points:
                n = len(points)
            else:
                n = n_points
            ind = numpy.argpartition(values, -n)[-n:]
            src_points.append(points[ind,:3])
    if src_points:
        src_points = numpy.concatenate(src_points)
        trg_points = src_points - numpy.asarray(offset)
        src_points *= 2**miplvl
        trg_points *= 2**miplvl
    else:
        trg_points = []
    tspec = [{"p_tile":src_tile,"q_tile":trg_tile,"p_pts":src_points.astype(int),"q_pts":trg_points.astype(int)}]
    save_pointmatch_file(tspec,output_file)
    
    
def extract_points_from_tiles(src_tile,trg_tile,output_file,miplvl,roi_file,method,blob_kwargs=None,points_kwargs=None):
    points_kwargs = ({} if points_kwargs is None else points_kwargs)
    if roi_file is None:
        roi_list = []
    else:
        with open(roi_file,'r') as f:
            roi_list = json.load(f)
    save_extracted_pointmatch_file(src_tile,trg_tile,output_file,miplvl,roi_list,method,blob_kwargs,**points_kwargs)
            

class BlobDetectionParameters(argschema.schemas.DefaultSchema):
    method = argschema.fields.Str(required=False,default=None)
    blob_kwargs = argschema.fields.Dict(required=False,default=None)


class PointExtractionParameters(argschema.schemas.DefaultSchema):
    n_points = argschema.fields.Int(required=False,default=1)


class ExtractPointsParameters(argschema.ArgSchema,
                              BlobDetectionParameters,
                              PointExtractionParameters):
    p_tile = argschema.fields.Str(required=True)
    q_tile = argschema.fields.Str(required=True)
    output_file = argschema.fields.Str(required=True)
    mip_lvl = argschema.fields.Int(required=False,default=0)
    roi_file = argschema.fields.Str(required=False,default=None)
    

class ExtractPointsFromTilePair(argschema.ArgSchemaParser):
    default_schema = ExtractPointsParameters
    
    def run(self):
        extract_points_from_tiles(self.args['p_tile'],
                                  self.args['q_tile'],
                                  self.args['output_file'],
                                  self.args['mip_lvl'],
                                  self.args['roi_file'],
                                  self.args['method'],
                                  blob_kwargs=self.args['blob_kwargs'])

if __name__ == "__main__":
    mod = ExtractPointsFromTilePair()
    mod.run()