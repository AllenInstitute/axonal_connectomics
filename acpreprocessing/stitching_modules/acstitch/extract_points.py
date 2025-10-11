# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 11:19:46 2025

@author: kevint
"""

import argschema
from skimage.feature import blob_log,blob_dog,blob_doh
from acpreprocessing.stitching_modules.acstitch.zarrutils import get_zarr_array

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
    
    data = get_roi(zarray,z_range,y_range,x_range)
    blobs = detect_blobs(data,method,**blob_kwargs)
    return blobs


def extract_points(tile_path,miplvl,method,roi_list,blob_kwargs):
    # input tile path to run method with blob kwargs at mip level
    # return 
    zarray = get_zarr_array(tile_path,miplvl=miplvl)
    blobs = []
    for roi in roi_list:
        blobs_roi = detect_blobs_roi(zarray,roi["z"],roi["y"],roi["x"],method=method,blob_kwargs=blob_kwargs)
        blobs.append(blobs_roi)
    return blobs
    

def save_points_to_file(tile_path,output_file,miplvl,method,roi_list,blob_kwargs):
    points = extract_points(tile_path,miplvl,method,blob_kwargs)
    #save points


class BlobDetectionParameters(argschema.schemas.DefaultSchema):
    method = argschema.fields.Str(required=False,default=None)
    blob_kwargs = argschema.fields.Dict(required=False,default=None)


class ExtractPointsParameters(argschema.ArgSchema,BlobDetectionParameters):
    tile_path = argschema.fields.Str(required=True)
    output_file = argschema.fields.Str(required=True)
    mip_lvl = argschema.fields.Int(required=False,default=0)
    

class ExtractPointsFromTile(argschema.ArgSchemaParser):
    default_schema = ExtractPointsParameters
    
    def run(self):
        extract_points(self.args['tile_path'],
                       self.args['output_file'],
                         self.args['mip_lvl'],
                         self.args['method'],
                         blob_kwargs=self.args['blob_kwargs'])

if __name__ == "__main__":
    mod = ExtractPointsFromTile()
    mod.run()