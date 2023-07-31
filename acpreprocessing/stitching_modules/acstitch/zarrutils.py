import os
import zarr

def get_zarr_group(zpath,grpname):
    # key to working with zarr files
    # group contains mip datasets and dataset attributes
    zf = zarr.open(zpath)
    return zf[grpname]

def get_group_from_src(srcpath):
    # returns zarr group given a neuroglancer source path
    # used to get datasets from neuroglancer layer json
    pathout = 'zarr://http://bigkahuna.corp.alleninstitute.org/ACdata' # Url for ACdata for NG hosted on BigKahuna
    pathin = 'J:' # Local path to ACdata
    s = os.path.split(srcpath.replace(pathout,pathin))
    return get_zarr_group(zpath=s[0],grpname=s[1])