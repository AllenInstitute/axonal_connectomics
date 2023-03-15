"""Pixel shift deskew
implements (chunked) pixel shifting deskew
skew_dims_zyx = dimensions of skewed (input) tiff data (xy are camera coordinates, z is tiff chunk #size, xz define skewed plane and y is non-skewed axis)
stride = number of camera (x) pixels to shift onto a sample (z') plane (sample z dim = camera x #dim/stride)
deskewFlip = flip volume (reflection, parity inversion)
dtype = datatype of input data

NOTE: must be run sequentially as each tiff chunk contains data for the next deskewed block #retained in self.slice1d except for the final chunk which should form the rhomboid edge
"""

import numpy as np


def psdeskew_kwargs(skew_dims_zyx, stride=1, deskewFlip=False, dtype='uint16', crop_factor=1, **kwargs):
    """get keyword arguments for deskew_block

    Parameters
    ----------
    skew_dims_zyx : tuple of int
        dimensions of raw data array block to be deskewed
    stride : int
        number of camera pixels per deskewed sampling plane (divides z resolution)
    deskewFlip : bool
        flip data blocks before deskewing
    dtype : str
        datatype for deskew output
    crop_factor : float
        reduce y dimension according to ydim*crop_factor < ydim

    Returns
    ----------
    dict of parameters representing pixel deskew operation for deskew_block
    """
    sdims = skew_dims_zyx
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
              'flip': deskewFlip,
              'dtype': dtype,
              'chunklength': blockx
              }
    return kwargs


def deskew_block(blockData, n, dsi, si, slice1d, blockdims, subblocks, flip, dtype, chunklength, *args, **kwargs):
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
    ndarray of pixel shifted deskewed data ordered (z,y,x) by sample axes 
    """
    subb = subblocks
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


def reshape_joined_shapes(joined_shapes, stride, blockdims, *args, **kwargs):
    deskewed_shape = (int(np.ceil(joined_shapes[0]/(blockdims[2]/stride))*blockdims[2]),
                      blockdims[1],
                      blockdims[0])
    return deskewed_shape

# def options_from_str(idstr):
#     """config lookup for deskew options by keystring
#     """
#     if idstr == 'ispim2':
#         options = {'stride':2,
#                    'deskewFlip':True,
#                    'dtype':'uint16',
#                    'crop_factor':0.5}
#     else:
#         options = {}

#     return options
