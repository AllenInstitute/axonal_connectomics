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

def random_points(data,n_points=100,seed=None):
    # input 3d image data array
    # return up to n_points random coordinates where data is nonzero
    nz_coords = numpy.argwhere(data != 0)
    if len(nz_coords) == 0:
        return numpy.empty((0,data.ndim))
    rng = numpy.random.default_rng(seed)
    n = min(n_points,len(nz_coords))
    idx = rng.choice(len(nz_coords),size=n,replace=False)
    return nz_coords[idx].astype(float)


def detect_blobs(data,method=None,**kwargs):
    # input 3d image data array
    # return blobs detected by method, or random nonzero points if method == "random"

    if method == "random":
        return random_points(data,**kwargs)

    p1 = numpy.percentile(data, 96)
    data =  data>p1
    
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
        print(zarray,z_range,y_range,x_range)
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


def evenly_spaced_points(bounds,n_points):
    # bounds: iterable of (lo,hi) per axis, e.g. [(z0,z1),(y0,y1),(x0,x1)]
    # return up to n_points points laid out on a regular grid spanning bounds
    bounds = numpy.asarray(bounds,dtype=float)
    dims = bounds[:,1] - bounds[:,0]

    if n_points <= 0 or numpy.any(dims <= 0):
        return numpy.empty((0,bounds.shape[0]))

    # pick a per-axis point count roughly proportional to that axis's length,
    # such that the product of counts is close to n_points
    volume = numpy.prod(dims)
    density = (n_points/volume) ** (1.0/len(dims))
    counts = numpy.maximum(1,numpy.round(dims*density)).astype(int)

    # trim any overshoot from rounding
    while numpy.prod(counts) > n_points and numpy.any(counts > 1):
        i = numpy.argmax(counts)
        counts[i] -= 1

    axes = []
    for (lo,hi),c in zip(bounds,counts):
        step = (hi-lo)/c
        axes.append(lo + step*(numpy.arange(c)+0.5))  # cell-centered grid

    mesh = numpy.meshgrid(*axes,indexing='ij')
    return numpy.stack([m.ravel() for m in mesh],axis=-1).astype('int32')


def compute_overlap_bounds(src_shape,trg_shape,offset):
    # src_shape,trg_shape: (z,y,x) array shapes at the given mip level
    # offset: translation such that q_pts = p_pts - offset (src -> trg)
    # return overlap bounds expressed in the src tile's local coordinate frame
    src_shape = numpy.asarray(src_shape[-3:])
    trg_shape = numpy.asarray(trg_shape[-3:])
    offset = numpy.asarray(offset)

    lo = numpy.maximum(0,offset)
    hi = numpy.minimum(src_shape,offset+trg_shape)
    return list(zip(lo,hi))


def extract_grid_points(src_zarray,trg_zarray,offset,n_points=100):
    # input src/trg zarr arrays (already sliced to a mip level) and the offset between them
    # return up to n_points points evenly distributed across their overlapping volume,
    # expressed in src local coordinates
    bounds = compute_overlap_bounds(src_zarray.shape[-3:],trg_zarray.shape[-3:],offset)
    return evenly_spaced_points(bounds,n_points)


def save_extracted_pointmatch_file(src_tile,trg_tile,output_file,miplvl,roi_list,method,blob_kwargs=None, q_offset=None, n_points=100):
    try:
        p_md = ZarrV3Metadata(zgroup=get_zarr_group(src_tile))
        q_md = ZarrV3Metadata(zgroup=get_zarr_group(trg_tile))
        offset = numpy.divide(q_md.get_coordinate_translation(),q_md.mip_voxel_dims(miplvl))-numpy.divide(p_md.get_coordinate_translation(),p_md.mip_voxel_dims(miplvl))
    except:
        offset = numpy.array([0,0,0])
    if q_offset:
        offset = offset + numpy.array(q_offset)
    print(offset)
    zarray = get_zarr_array(src_tile,miplvl=miplvl)

    if method == "grid":
        # evenly distributed points across the whole overlapping volume; no ROIs, no blob detection
        trg_zarray = get_zarr_array(trg_tile,miplvl=miplvl)
        src_points = extract_grid_points(zarray,trg_zarray,offset,n_points=n_points)
        new_points = []
        for point in src_points:
            x,y,z = point
            if zarray[0,0,x,y,z] > 0:
                new_points.append(point)
        src_points = numpy.array(new_points)
        if len(src_points) == 0:
            print("no overlap between tiles!!")
    else:
        blob_kwargs = ({} if blob_kwargs is None else dict(blob_kwargs))
        blob_kwargs = ({} if blob_kwargs is None else dict(blob_kwargs))
        if method == "random":
            blob_kwargs.setdefault("n_points", n_points)

        src_points_rois,values_rois = extract_points(zarray,roi_list,method,blob_kwargs)
        collected = []
        for points,values in zip(src_points_rois,values_rois):
            if len(points) > 0:
                n = min(len(points),n_points)
                ind = numpy.argpartition(values, -n)[-n:]
                collected.append(points[ind,:3])
        src_points = numpy.concatenate(collected) if collected else numpy.empty((0,3))

    if len(src_points) > 0:
        trg_points = src_points - numpy.asarray(offset)
        src_points = src_points * 2**miplvl
        trg_points = trg_points * 2**miplvl
    else:
        trg_points = numpy.empty((0,3))

    #print(src_points)
    tspec = [{"p_tile":src_tile,"q_tile":trg_tile,"p_pts":src_points.astype(int),"q_pts":trg_points.astype(int)}]
    save_pointmatch_file(tspec,output_file)
    
    
def extract_points_from_tiles(src_tile,trg_tile,output_file,miplvl,roi_file,method,blob_kwargs=None,points_kwargs=None, q_offset=None, n_points=10):
    points_kwargs = ({} if points_kwargs is None else points_kwargs)
    #print(n_points)
    if roi_file is None:
        roi_list = []
    else:
        with open(roi_file,'r') as f:
            roi_list = json.load(f)
    save_extracted_pointmatch_file(src_tile,trg_tile,output_file,miplvl,roi_list,method,blob_kwargs,q_offset, n_points, **points_kwargs)
            

class BlobDetectionParameters(argschema.schemas.DefaultSchema):
    method = argschema.fields.Str(required=False,default=None,
        metadata={"description":"blob detection method ('log','dog','doh'), 'random' for random nonzero points, "
                                 "or 'grid' for points evenly distributed across the tile overlap (ignores roi_file)"})
    blob_kwargs = argschema.fields.Dict(required=False,default=None, allow_none=True)


class PointExtractionParameters(argschema.schemas.DefaultSchema):
    n_points = argschema.fields.Int(required=False,default=1)


class ExtractPointsParameters(argschema.ArgSchema,
                              BlobDetectionParameters,
                              PointExtractionParameters):
    p_tile = argschema.fields.Str(required=True)
    q_tile = argschema.fields.Str(required=True)
    output_file = argschema.fields.Str(required=True)
    mip_lvl = argschema.fields.Int(required=False,default=0)
    roi_file = argschema.fields.Str(required=False,default=None,allow_none=True)
    q_offset = argschema.fields.List(argschema.fields.Int(),
                                      required=False,
                                      default=[0,0,0],
                                      cli_as_single_argument=True)
    
class ExtractPointsFromTilePair(argschema.ArgSchemaParser):
    default_schema = ExtractPointsParameters
    
    def run(self):
        extract_points_from_tiles(self.args['p_tile'],
                                  self.args['q_tile'],
                                  self.args['output_file'],
                                  self.args['mip_lvl'],
                                  self.args['roi_file'],
                                  self.args['method'],
                                  blob_kwargs=self.args['blob_kwargs'],
                                  q_offset=self.args['q_offset'],
                                  n_points=self.args['n_points'])
if __name__ == "__main__":
    mod = ExtractPointsFromTilePair()
    mod.run()