import pathlib
import zarr
import json
import numpy

def get_zarr_array(zpath,grpname=None,miplvl=0):
    zg = get_zarr_group(zpath,grpname)
    return zg[f"{miplvl}"]
    

def get_zarr_group(zpath,grpname=None):
    # key to working with zarr files
    # group contains mip datasets and dataset attributes
    zf = zarr.open(zpath)
    if not grpname is None:
        return zf[grpname]
    else:
        return zf


def get_group_from_src(srcpath,
                       outpath='zarr://http://bigkahuna.corp.alleninstitute.org/ACdata', # Url for ACdata for NG hosted on BigKahuna
                       inpath = 'J:'):
    # returns zarr group given a neuroglancer source path
    # used to get datasets from neuroglancer layer json
    p = pathlib.Path(srcpath.replace(outpath,inpath))
    if p.exists():
        return get_zarr_group(p.parent,p.name)
    else:
        print(str(p) + " not found!")
        return None
    

def get_src_from_json(sourcejson,plane,tile):
    with open(sourcejson,'r') as f:
        js = json.load(f)
    srcList = js[plane]['sources']
    ind = [s.split("_")[-1] for s in srcList].index(tile)
    return srcList[ind]


class ZarrV3Metadata:
    
    def __init__(self,zgroup=None):
        self.zgroup = zgroup
        if not self.zgroup is None:
            self.attrs = self.zgroup.attrs
        else:
            self.attrs = None
    
    def mip_voxel_dims(self,miplvl=0):
        dims = self.attrs["multiscales"][0]["datasets"][miplvl]["coordinateTransformations"][0]["scale"][2:]
        return numpy.array(dims)

    def get_coordinate_translation(self):
        trans = self.attrs["multiscales"][0]["coordinateTransformations"][0]["translation"][2:]
        return numpy.array(trans)