import pathlib
import zarr

def get_zarr_group(zpath,grpname):
    # key to working with zarr files
    # group contains mip datasets and dataset attributes
    zf = zarr.open(zpath)
    return zf[grpname]

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