from scipy import ndimage
import itertools
from joblib import Parallel, delayed, parallel_config, dump, load
import tinybrain
from acpreprocessing.flatten.flatten_utils import find_surface

def resample_dim(shape, zfactor=(2,2,2), sample='up'):
    if sample == 'up':
        out_shape = np.array(shape)*np.array(zfactor)
    elif sample =='down':
        out_shape = np.round(np.array(shape)/np.array(zfactor)).astype(int)
    return out_shape

def upsample_array_parallel(in_arr, out_arr, upfactor=(2,2,2), chunk_size=(64,64,64),  n_jobs=1):
    def upsample(start, end, upfactor):
        if isinstance(in_arr, ts.TensorStore):
            arr = in_arr[start[0]:end[0],start[1]:end[1],start[2]:end[2]].read().result()
        else:
            arr = in_arr[start[0]:end[0],start[1]:end[1],start[2]:end[2]]
        rs_image = ndimage.zoom(arr, upfactor)
        start, end = np.array(start)*np.array(upfactor), np.array(end)*np.array(upfactor)
        if isinstance(out_arr, ts.TensorStore):
            out_arr[start[0]:end[0],start[1]:end[1],start[2]:end[2]].write(rs_image).result()
        else:
            out_arr[start[0]:end[0],start[1]:end[1],start[2]:end[2]] = rs_image

    #confirm output dimensions
    out_dim = np.array(in_arr.shape)*np.array(upfactor)
    check = np.array(out_arr.shape) == out_dim
    if not check.all():
        error_string = "The out array dimensions conflict with the upsample factor. They should be {} ".format(out_dim)
        raise ValueError(error_string)
    
    dx,dy,dz = in_arr.shape
    xch, ych, zch = chunk_size
    sind_x, sind_y, sind_z = list(range(0,dx,xch)), list(range(0,dy,ych)), list(range(0,dz,zch))
    eind_x, eind_y, eind_z = [x + xch for x in sind_x], [x + ych for x in sind_y], [x + zch for x in sind_z]
    eind_x, eind_y, eind_z = [dx if ele > dx else ele for ele in eind_x], [dy if ele > dy else ele for ele in eind_y], [dz if ele > dz else ele for ele in eind_z]
    comb1 = list(itertools.product(sind_x,sind_y,sind_z))
    comb2 = list(itertools.product(eind_x,eind_y,eind_z))
    del sind_x, sind_y, sind_z, eind_x, eind_y, eind_z
    
    #run upsampling in parallel
    with parallel_config(backend="loky", inner_max_num_threads=2):
        results = Parallel(n_jobs=n_jobs)(delayed(upsample)(start=start, end=end, upfactor=upfactor) for start, end in zip(comb1,comb2))
        
def downsample_array_parallel(in_arr, out_arr, downfactor=(2,2,2), chunk_size=(64,64,64),  n_jobs=1):
    def downsample(start, end, downfactor):
        if isinstance(in_arr, ts.TensorStore):
            arr = in_arr[start[0]:end[0],start[1]:end[1],start[2]:end[2]].read().result()
        else:
            arr = in_arr[start[0]:end[0],start[1]:end[1],start[2]:end[2]]
        rs_image = tinybrain.downsample_with_averaging(arr, factor=downfactor)[0]
        start, end = resample_dim(start, zfactor=downfactor, sample='down'), resample_dim(end, zfactor=downfactor, sample='down')
        if isinstance(out_arr, ts.TensorStore):
            out_arr[start[0]:end[0],start[1]:end[1],start[2]:end[2]].write(rs_image).result()
        else:
            out_arr[start[0]:end[0],start[1]:end[1],start[2]:end[2]] = rs_image

    #confirm output dimensions
    out_dim = np.round(np.array(in_arr.shape)/np.array(downfactor))
    check = np.array(out_arr.shape) == out_dim
    if not check.all():
        error_string = "The out array dimensions conflict with the downsample factor. They should be {} ".format(out_dim)
        raise ValueError(error_string)
    
    dx,dy,dz = in_arr.shape
    xch, ych, zch = chunk_size
    sind_x, sind_y, sind_z = list(range(0,dx,xch)), list(range(0,dy,ych)), list(range(0,dz,zch))
    eind_x, eind_y, eind_z = [x + xch for x in sind_x], [x + ych for x in sind_y], [x + zch for x in sind_z]
    eind_x, eind_y, eind_z = [dx if ele > dx else ele for ele in eind_x], [dy if ele > dy else ele for ele in eind_y], [dz if ele > dz else ele for ele in eind_z]
    comb1 = list(itertools.product(sind_x,sind_y,sind_z))
    comb2 = list(itertools.product(eind_x,eind_y,eind_z))
    del sind_x, sind_y, sind_z, eind_x, eind_y, eind_z

    #run upsampling in parallel
    with parallel_config(backend="loky", inner_max_num_threads=2):
        results = Parallel(n_jobs=n_jobs)(delayed(downsample)(start=start, end=end, downfactor=downfactor) for start, end in zip(comb1,comb2))

def create_mask(top, bottom, shape, zarr=False):
    """
    :param top: 2d array, int, indices of top surface
    :param bottom: 2d array, int, indices of bottom surface
    :param shape: 3d tuple, dimensions of array, ZYX
    :return bot: 2d array, same size as each plane in img, z index of bottom surface
    """
    total = 0
    if zarr==True:
        mask = create_tensor(os.getcwd()+"/out_mask.zarr", arr_shape=shape, dtype='uint8', driver='zarr3', fill_value=0,
                            codec={"metadata": {"codecs": [{"name": "gzip", "configuration": {"level": 9}}]}})
    else:
        mask = np.zeros(shape).astype('uint8')

    writes = []
    for y in range(shape[1]):
        block = np.zeros([shape[0],1,shape[2]], dtype='uint8')
        for x in range(shape[2]):
            zmin,zmax = top[y][x], bottom[y][x]
            total += zmax-zmin
            block[zmin:zmax,:,x] = 1
        if zarr == True:
            mask[:,y:y+1,:].write(block).result()
        else:
            mask[:,y:y+1,:] = block
    
    print("% Pixels Masked:",100-(total/(shape[0]*shape[1]*shape[2]))*100)
    return mask
