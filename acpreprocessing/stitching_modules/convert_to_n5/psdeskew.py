"""Pixel shift deskew
implements (chunked) pixel shifting deskew
skew_dims_zyx = dimensions of skewed (input) tiff data (xy are camera coordinates, z is tiff chunk #size, xz define skewed plane and y is non-skewed axis)
stride = number of camera (x) pixels to shift onto a sample (z') plane (sample z dim = camera x #dim/stride)
deskewFlip = flip volume (reflection, parity inversion)
dtype = datatype of input data

NOTE: must be run sequentially as each tiff chunk contains data for the next deskewed block #retained in self.slice1d except for the final chunk which should form the rhomboid edge
"""

import numpy as np


def psdeskew_kwargs(skew_dims_zyx, deskew_stride=1, deskew_flip=False, deskew_transpose=False, deskew_crop=1, dtype='uint16', **kwargs):
    """get keyword arguments for deskew_block

    Parameters
    ----------
    skew_dims_zyx : tuple of int
        dimensions of raw data array block to be deskewed
    stride : int
        number of camera pixels per deskewed sampling plane (divides z resolution)
    deskewFlip : bool
        flip data blocks before deskewing
    deskewTranspose : bool
        transpose x,y axes before deskewing
    dtype : str
        datatype for deskew output
    crop_factor : float
        reduce y dimension according to ydim*crop_factor < ydim

    Returns
    ----------
    kwargs : dict
        parameters representing pixel deskew operation for deskew_block
    """
    # if deskew_transpose:
    #     skew_dims_zyx = (skew_dims_zyx[0],skew_dims_zyx[2],skew_dims_zyx[1])
    sdims = skew_dims_zyx
    crop_factor = deskew_crop
    stride = deskew_stride
    ydim = int(sdims[1]*crop_factor)
    blockdims = (int(sdims[2]/stride), ydim, stride*sdims[0])
    subblocks = int(np.ceil((sdims[2]+stride*sdims[0])/(stride*sdims[0])))
    # print(subblocks)
    blockx = sdims[0]
    dsi = []
    si = []
    for i_block in range(subblocks):
        sxv = []
        szv = []
        for sz in range(blockx):
            sxstart = i_block*stride*blockx-stride*sz
            sxend = (i_block+1)*stride*blockx-stride*sz
            if sxstart < 0:
                sxstart = 0
            if sxend > sdims[2]:
                sxend = sdims[2]
            sx = np.arange(sxstart, sxend)
            sxv.append(sx)
            szv.append(sz*np.ones(sx.shape, dtype=sx.dtype))
        sxv = np.concatenate(sxv)
        szv = np.concatenate(szv)
        dsx = sxv + stride*szv - i_block*stride*blockx
        dsz = np.floor(sxv/stride).astype(int)
        dsi.append(np.ravel_multi_index(
            (dsz, dsx), (blockdims[0], blockdims[2])))
        si.append(np.ravel_multi_index((szv, sxv), (sdims[0], sdims[2])))
    kwargs = {'dsi': dsi,
              'si': si,
              'slice1d': np.zeros((subblocks, blockdims[1], blockdims[2]*blockdims[0]), dtype=dtype),
              'blockdims': blockdims,
              'subblocks': subblocks,
              'flip': deskew_flip,
              'transpose': deskew_transpose,
              'dtype': dtype,
              'chunklength': blockx,
              'stride': stride
              }
    return kwargs


def calculate_skewed_indices(zi,yi,xi,s):
    # convert input block voxel indices into skewed data space
    # edge blocks may have negative or out-of-bounds skewed indices
    xs = s*zi + xi % s
    ys = yi
    zs = xi // s - zi
    return zs,ys,xs


def get_deskewed_block(blockdims,dataset,start,end,stride):
    # create output block and get flattened indices
    blockdata = np.zeros(blockdims,dtype=dataset.dtype)
    zb,yb,xb = np.meshgrid(*[range(d) for d in blockdims],indexing="ij")
    fb = np.ravel_multi_index((zb,yb,xb),blockdims)
    # get indices of voxel data for input dataset
    sdims = dataset.shape
    zi,yi,xi = np.meshgrid(*[range(s,e) for s,e in zip(start,end)],indexing="ij")
    zs,ys,xs = calculate_skewed_indices(zi,yi,xi,stride)
    fi = np.ravel_multi_index((zs,ys,xs),sdims,mode='clip').flatten()
    # filter out-of-bounds voxels
    r = (fi > 0) & (fi < np.prod(sdims)-1)
    fb = fb[r]
    fi = fi[r]
    # assign input to output
    blockdata[fb] = dataset[fi]
    return blockdata


def calculate_first_chunk(dataset_shape,chunk_size,x_index,stride):
    first_chunk = x_index*stride
    first_slice = int(x_index*chunk_size[0]*stride) #% chunk_size[0]
    return first_chunk,first_slice


def deskew_block(blockData, n, dsi, si, slice1d, blockdims, subblocks, flip, transpose, dtype, chunklength, *args, **kwargs):
    """deskew a data chunk in sequence with prior chunks

    Parameters
    ----------
    blockData : numpy.ndarray
        block of raw (nondeskewed) data to be deskewed
    n : int
        current iteration in block sequence (must be run sequentially)
    dsi : numpy.ndarray
        deskewed indices for reslicing flattened data
    si : numpy.ndarray
        skewed indices for sampling flattened raw data
    slice1d : numpy.ndarray
        flattened data from previous iteration containing data for next deskewed block
    blockdims : tuple of int
        dimensions of output block
    subblocks : int
        number of partitions of input block for processing - likely not necessary
    flip : bool
        deskew flip
    dtype : str
        datatype
    chunklength : int
        number of slices expected for raw data block (for zero filling)

    Returns
    ----------
    block3d : numpy.ndarray
        pixel shifted deskewed data ordered (z,y,x) by sample axes 
    """
    # if transpose:
    #     blockData = blockData.transpose((0,2,1))
    subb = subblocks
    # subb = 5
    block3d = np.zeros(blockdims, dtype=dtype)
    zdim = block3d.shape[0]
    ydim = block3d.shape[1]
    xdim = block3d.shape[2]
    # crop blockData if needed
    if blockData.shape[1] > ydim:
        y0 = int(np.floor((blockData.shape[1]-ydim)/2))
        y1 = int(np.floor((blockData.shape[1]+ydim)/2))
        blockData = blockData[:, y0:y1, :]
    #print('deskewing block ' + str(n) + ' with shape ' + str(blockData.shape))
    if blockData.shape[0] < chunklength:
        #print('block is short, filling with zeros')
        blockData = np.concatenate((blockData, np.zeros(
            (int(chunklength-blockData.shape[0]), blockData.shape[1], blockData.shape[2]))))
    order = (np.arange(subb)+n) % subb
    for y in range(ydim):
        for i, o in enumerate(order):
            # flip stack axis 2 for ispim2
            s = -1 if flip else 1
            slice1d[o, y, :][dsi[i]] = blockData[:, y, ::s].ravel()[si[i]]
        block3d[:, y, :] = slice1d[n % subb, y, :].reshape((zdim, xdim))
        slice1d[n % subb, y, :] = 0
    return block3d


def reshape_joined_shapes(joined_shapes, stride, blockdims, transpose=None, **kwargs):
    """get dimensions of deskewed joined shapes from skewed joined shapes

    Parameters
    ----------
    joined_shapes : tuple of int
        shape of 3D array represented by concatenating mimg_fns
    stride : int
        number of camera pixels per deskewed sampling plane (divides z resolution)
    blockdims : tuple of int
        dimensions of output block

    Returns
    ----------
    deskewed_shape : tuple of int
        shape of deskewed 3D array represented by joined_shapes
    """
    if not transpose is None:
        axes = transpose
    else:
        axes = (0,1,2)
    # deskewed_shape = (int(np.ceil(joined_shapes[axes[0]]/(blockdims[axes[0]]/stride))*blockdims[axes[0]]),
    #                   joined_shapes[axes[1]],
    #                   joined_shapes[axes[2]])
    deskewed_shape = (int(np.ceil(joined_shapes[axes[0]]/(blockdims[axes[0]]/stride))*blockdims[axes[0]]),
                      joined_shapes[axes[1]],
                      int(np.ceil((joined_shapes[axes[0]] + joined_shapes[axes[2]]*stride)/(blockdims[axes[0]]))*blockdims[axes[0]]))
    return deskewed_shape
