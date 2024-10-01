#!/usr/bin/env python

import concurrent.futures
import dataclasses
#import itertools
# import math
#import pathlib

# import imageio
#from natsort import natsorted
import numpy
# import skimage

import hdf5plugin # this needs to be imported
import h5py
from imaris_ims_file_reader.ims import ims
import zarr
from numcodecs import Blosc
import argschema

import acpreprocessing.stitching_modules.convert_to_n5.psdeskew as psd
from acpreprocessing.stitching_modules.convert_to_n5.tiff_to_ngff import downsample_array,mip_level_shape,omezarr_attrs,NGFFGroupGenerationParameters,TiffToNGFFValueError


@dataclasses.dataclass
class MIPArray:
    lvl: int
    array: numpy.ndarray
    start: tuple # list of indices (int)
    end: tuple # list of indices (int)


def dswrite_block(ds, start, end, arr, silent_overflow=True):
    """write array arr into array-like n5 dataset defined by
    start and end point 3-tuples.

    Parameters
    ----------
    ds : z5py.dataset.Dataset (n5 3D) or zarr.dataset (zarr 5D)
        array-like dataset to fill in
    start : tuple[int]
        start index along 0 axis of ds to fill in
    end : tuple[int]
        end index along 0 axis of ds to fill in
    arr : numpy.ndarray
        array from which ds values will be taken
    silent_overflow : bool, optional
        whether to shrink the end index to match the
        shape of ds (default: True)
    """
    start = list(start)
    end = list(end)
    if len(ds.shape) == 5:  # dataset dimensions should be 3 or 5
        for i in range(3):
            if end[i] >= ds.shape[2+i] and silent_overflow:
                end[i] = ds.shape[2+i]
        #if end > start:
        ds[0, 0, start[0]:end[0], start[1]:end[1], start[2]:end[2]] = arr[:(end[0] - start[0]), :(end[1] - start[1]), :(end[2] - start[2])]
    elif len(ds.shape) == 3:
        for i in range(3):    
            if end[i] >= ds.shape[i] and silent_overflow:
                end[i] = ds.shape[i]
        #if end > start:
        ds[start[0]:end[0], start[1]:end[1], start[2]:end[2]] = arr[:(end[0] - start[0]), :(end[1] - start[1]), :(end[2] - start[2])]


def write_mips(zgrp,miparrs):
    for miparr in miparrs:
        dswrite_block(ds=zgrp[miparr.lvl],start=miparr.start,end=miparr.end,arr=miparr.array)


def iterate_numpy_blocks_from_dataset(
        dataset, nblocks, numslice=0, chunknum=-1, block_size=None, pad=True, deskew_kwargs={}, *args, **kwargs):
    """iterate over a contiguous hdf5 daataset as chunks of numpy arrays

    Parameters
    ----------
    dataset : hdf5 dataset
        imageio-compatible name inputs to be opened as multi-images
    nblocks : tuple[int]
        number of 2d arrays from mimgfns included per chunk
    pad : bool, optional
        whether to extend final returned chunks with zeros

    Yields
    ------
    arr : numpy.ndarray
        3D numpy array representing a consecutive chunk of 2D arrays
    """
    
    if numslice == 1:
        print("1 slice: using chunk x = " + str(chunknum))
        nchunks = (nblocks[0],nblocks[1],1)
        test = True
    else:
        nchunks = nblocks
        test = False
    dshape = dataset.shape[2:]
    if deskew_kwargs:
        chunk_size = (deskew_kwargs["chunklength"],block_size[1],block_size[2]*deskew_kwargs["stride"])
        if deskew_kwargs["transpose"]:
            chunk_size = (chunk_size[0],chunk_size[2],chunk_size[1])
            dshape = (dshape[0],dshape[2],dshape[1])
        print("chunk size: " + str(chunk_size))
    for i in range(numpy.prod(nchunks)):#,*args,**kwargs):
        chunk_tuple = numpy.unravel_index(i,tuple(nchunks),order='F')
        if test:
            chunk_tuple = (chunk_tuple[0], chunk_tuple[1], chunknum)
        # deskew level 0 data blocks
        if deskew_kwargs:
            if deskew_kwargs["transpose"]:
                chunk_tuple = (chunk_tuple[0],chunk_tuple[2],chunk_tuple[1])
            print(str(chunk_tuple))
            if chunk_tuple[0] == 0:
                chunk_index = 0
                deskew_kwargs["slice1d"][...] = 0
                first_z,first_slice = psd.calculate_first_chunk(chunk_size=chunk_size,x_index=(nblocks[2] - chunk_tuple[1] - 1),stride=deskew_kwargs["stride"])
                print(str(first_z) + "," + str(first_slice))
            if chunk_tuple[0] < first_z or chunk_tuple[0]*chunk_size[0] - first_slice >= dshape[0]:
                arr = numpy.zeros(block_size,dtype=dataset.dtype)
            else:
                chunk_start = numpy.array([t*s for t,s in zip(chunk_tuple,chunk_size)])
                chunk_end = chunk_start + numpy.array(chunk_size)
                if chunk_start[0] < first_slice:
                    chunk_end[0] = chunk_size[0] - (first_slice - chunk_start[0])
                    chunk = numpy.zeros(chunk_size,dtype=dataset.dtype)
                    zdata = numpy.squeeze(numpy.asarray(dataset[0,0,:chunk_end[0],chunk_start[1]:chunk_end[1],chunk_start[2]:chunk_end[2]]))
                    print("data dimension is " + str(zdata.shape) + " max is " + str(numpy.max(zdata)))
                    chunk[first_slice-chunk_start[0]:] = zdata
                else:
                    chunk_start[0] -= first_slice
                    chunk_end[0] -= first_slice
                    # if chunk_end[0] >= dshape[0]:
                    #     chunk = numpy.zeros(chunk_size,dtype=dataset.dtype)
                    #     chunk[:dshape[0]-chunk_start[0]] = dataset[chunk_start[0]:,chunk_start[1]:chunk_end[1],chunk_start[2]:chunk_end[2]]
                    # else:
                    print(str(chunk_start[0]))
                    zdata = numpy.squeeze(numpy.asarray(dataset[0,0,chunk_start[0]:chunk_end[0],chunk_start[1]:chunk_end[1],chunk_start[2]:chunk_end[2]]))
                    print("data dimension is " + str(zdata.shape) + " max is " + str(numpy.max(zdata)))
                    chunk = zdata
                if any([sh<sz for sh,sz in zip(chunk.shape,chunk_size)]):
                    print(str(chunk.shape) + " is small for" + str(chunk_size) + ": filling with zeros")
                    temp_chunk = numpy.zeros(chunk_size,dtype=chunk.dtype)
                    temp_chunk[:chunk.shape[0],:chunk.shape[1],:chunk.shape[2]] = chunk
                    chunk = temp_chunk
                if deskew_kwargs["transpose"]:
                    chunk = chunk.transpose((0,2,1))
                arr = numpy.flip(
                    numpy.transpose(
                        psd.deskew_block(
                            chunk,
                            chunk_index,
                            **deskew_kwargs), 
                        (2, 1, 0)),
                    axis=2)
            chunk_index += 1
        else:
            if chunk_tuple[0] == 0:
                print(str(chunk_tuple))
            block_start = [chunk_tuple[k]*block_size[k] for k in range(3)]
            block_end = [block_start[k] + block_size[k] for k in range(3)]
            arr = numpy.squeeze(numpy.asarray(dataset[0,0,block_start[0]:block_end[0],block_start[1]:block_end[1],block_start[2]:block_end[2]]))
        if any([arr.shape[k] != block_size[k] for k in range(3)]):
            print(str(arr.shape) + "is small for " + str(block_size))
            if pad:
                newarr = numpy.zeros(block_size,
                                      dtype=arr.dtype)
                newarr[:arr.shape[0], :arr.shape[1], :arr.shape[2]] = arr[:, :, :]
                yield newarr
            else:
                yield arr
        else:
            yield arr


def iterate_mip_levels_from_dataset(
        dataset, lvl, maxlvl, nblocks, block_size, downsample_factor,
        num_slice=0,chunknum=-1,downsample_method=None, lvl_to_mip_kwargs=None,
        interleaved_channels=1, channel=0, deskew_kwargs={}):
    """recursively generate MIPmap levels from an iterator of blocks

    Parameters
    ----------
    dataset : hdf5 dataset
        imageio-compatible name inputs to be opened as multi-images
    lvl : int
        integer mip level to generate
    block_size : int
        number of 2D arrays in chunk to process for a chunk at lvl
    slice_length : int
        number of 2D arrays to gather from tiff stacks
    downsample_factor : tuple of int
        integer downsampling factor for MIP levels
    downsample_method : str, optional
        downsampling method for
        acpreprocessing.utils.convert.downsample_stack_volume
    lvl_to_mip_kwargs :  dict, optional
        mapping of MIP level to kwargs used in MIPmap generation
    interleaved_channels : int
        number of channels interleaved in the tiff files (default 1)
    channel : int, optional
        channel from which interleaved data should be read (default 0)
    deskew_kwargs : dict, optional
        parameters for pixel shifting deskew

    Yields
    ------
    ma : acpreprocessing.stitching_modules.convert_to_n5.tiff_to_n5.MipArray
        object describing chunked array, MIP level of origin, and chunk indices
    """
    lvl_to_mip_kwargs = ({} if lvl_to_mip_kwargs is None
                          else lvl_to_mip_kwargs)
    mip_kwargs = lvl_to_mip_kwargs.get(lvl, {})
    block_index = 0
    if num_slice == 1:
        block_index = nblocks[0]*nblocks[1]*chunknum
    if lvl > 0:
        for ma in iterate_mip_levels_from_dataset(
                dataset, lvl-1, maxlvl, nblocks, block_size,
                downsample_factor,num_slice,chunknum, downsample_method,
                lvl_to_mip_kwargs, interleaved_channels=interleaved_channels,
                channel=channel, deskew_kwargs=deskew_kwargs):
            chunk = ma.array
            # throw array up for further processing
            yield ma
            # ignore if not parent resolution
            if ma.lvl != lvl-1:
                continue
            
            temp_arr = (downsample_array(
                    chunk, downsample_factor, dtype=chunk.dtype,
                    method=downsample_method, **mip_kwargs))
            chunk_start = tuple(int(ma.start[k]/downsample_factor[k]) for k in range(3))
            chunk_end = tuple(chunk_start[k] + temp_arr.shape[k] for k in range(3))
            yield MIPArray(lvl, temp_arr, chunk_start, chunk_end)
    else:
        # get level 0 chunks
        # block_size is the number of slices to read from tiffs
        for block in iterate_numpy_blocks_from_dataset(
                dataset, nblocks,num_slice=num_slice,chunknum=chunknum, block_size=block_size, pad=False,
                deskew_kwargs=deskew_kwargs,
                channel=channel):
            block_tuple = numpy.unravel_index(block_index,nblocks,order='F')
            block_start = tuple(block_tuple[k]*block_size[k] for k in range(3))
            block_end = tuple(block_start[k] + block.shape[k] for k in range(3))
            yield MIPArray(lvl, block, block_start, block_end)
            block_index += 1


def write_ims_to_zarr(
        ims_fn, output_n5, group_names, group_attributes=None, max_mip=0,
        mip_dsfactor=(2, 2, 2), chunk_size=(1, 1, 64, 64, 64),
        concurrency=10, slice_concurrency=1,
        compression="raw", dtype="uint16", lvl_to_mip_kwargs=None,
        interleaved_channels=1, channel=0, deskew_options=None, numchunks=0, chunknum=-1, **kwargs):
    """write a stack represented by an iterator of multi-image files as a zarr
    volume with ome-ngff metadata

    Parameters
    ----------
    ims_fn : str
        imageio-compatible name inputs to be opened as multi-images
    output_n5 : str
        output zarr directory
    group_names : list of str
        names of groups to generate within n5
    group_attributes : list of dict, optional
        attribute dictionaries corresponding to group with matching index
    max_mip : int
        maximum MIP level to generate
    mip_dsfactor : tuple of int
        integer downsampling factor for MIP levels
    chunk_size : tuple of int
        chunk size for n5 datasets
    concurrency : int
        total concurrency used for writing arrays to n5
        (python threads = concurrency // slice_concurrency)
    slice_concurrency : int
        threads used by z5py
    compression : str, optional
        compression for n5 (default: raw)
    dtype : str, optional
        dtype for n5 (default: uint16)
    lvl_to_mip_kwargs :  dict, optional
        mapping of MIP level to kwargs used in MIPmap generation
    interleaved_channels : int
        number of channels interleaved in the tiff files (default 1)
    channel : int, optional
        channel from which interleaved data should be read (default 0)
    deskew_options : dict, optional
        dictionary of parameters to run pixel shifting deskew (default None)
    """
    group_attributes = ([] if group_attributes is None else group_attributes)
    deskew_options = ({} if deskew_options is None else deskew_options)
    
    #f = h5py.File(ims_fn, 'r', rdcc_nbytes=512*1024**2)
    #dataset = f['DataSet']['ResolutionLevel 0']['TimePoint 0']['Channel 0']['Data']
    store = ims(ims_fn,ResolutionLevelLock=0,aszarr=True)
    dataset = zarr.open(store,mode="r")
    ims_chunk_size = store.chunks
    print("ims chunks: " + str(ims_chunk_size))
    # if deskew_options and deskew_options["deskew_transpose"]:
    #     dataset = dataset.transpose((0,2,1))
    #     print("transposed shape: " + str(dataset.shape))
    
    block_size = [256,512,512] #[ims_chunk_size[0]*2,ims_chunk_size[1]*32,ims_chunk_size[2]*8] #[128,2048,512] #[m*sz for m,sz in zip([2,2**max_mip,8],chunk_size[2:])]
    print("deskewed block size: " + str(block_size))
    
    # if numchunks < 1:
    joined_shapes = dataset.shape[2:]
    if deskew_options and deskew_options["deskew_transpose"]:
        # input dataset must be transposed
        joined_shapes = (joined_shapes[0],joined_shapes[2],joined_shapes[1])
    # else:
    #     if deskew_options and deskew_options["deskew_transpose"]:
    #         # input dataset must be transposed
    #         joined_shapes = (dataset.shape[2],dataset.shape[4],numchunks*block_size[2])#(dataset.shape[2],numchunks*block_size[1],dataset.shape[3])
    #     else:
    #         joined_shapes = (dataset.shape[2],dataset.shape[3],numchunks*block_size[2])#(dataset.shape[2],numchunks*block_size[1],dataset.shape[4])
    print("ims_to_ngff dataset shape:" + str(joined_shapes))

    if deskew_options and deskew_options["deskew_method"] == "ps":
        print(deskew_options)
        if not deskew_options["deskew_stride"] is None:
            stride = deskew_options["deskew_stride"]
        else:
            stride = 1
        slice_length = int(block_size[0]/stride)
        deskew_kwargs = psd.psdeskew_kwargs(skew_dims_zyx=(slice_length, block_size[1], block_size[2]*stride), # size of skewed chunk to iterate for deskewed block
                                            **deskew_options)
        joined_shapes = psd.reshape_joined_shapes(
            joined_shapes, stride, block_size)# transpose=(2,1,0))
        print("deskewed shape:" + str(joined_shapes))
    else:
        deskew_kwargs = {}

    # TODO DESKEW: does this generally work regardless of skew?

    workers = concurrency // slice_concurrency

    zstore = zarr.DirectoryStore(output_n5, dimension_separator='/')
    with zarr.open(zstore, mode='a') as f:
        mip_ds = {}
        # create groups with attributes according to omezarr spec
        if len(group_names) == 1:
            group_name = group_names[0]
            if group_name in f:
                g = f[f"{group_name}"]
            else:
                g = f.create_group(f"{group_name}")
            # try:
            #     g = f.create_group(f"{group_name}")
            # except KeyError:
            #     g = f[f"{group_name}"]
                
            if group_attributes:
                try:
                    attributes = group_attributes[0]
                except IndexError:
                    print('attributes error')
            else:
                attributes = {}

            if "pixelResolution" in attributes:
                if deskew_options:
                    attributes["pixelResolution"]["dimensions"][2] /= deskew_options["deskew_stride"]
                attributes = omezarr_attrs(
                    group_name, attributes["position"], attributes["pixelResolution"]["dimensions"], max_mip)
            if attributes:
                for k, v in attributes.items():
                    g.attrs[k] = v
        else:
            raise TiffToNGFFValueError("only one group name expected")
        scales = []

        # shuffle=Blosc.BITSHUFFLE)
        compression = Blosc(cname='zstd', clevel=1)
        for mip_lvl in range(max_mip + 1):
            mip_3dshape = mip_level_shape(mip_lvl, joined_shapes)
            ds_lvl = g.create_dataset(
                f"{mip_lvl}",
                chunks=chunk_size,
                shape=(1, 1, mip_3dshape[0], mip_3dshape[1], mip_3dshape[2]),
                compression=compression,
                synchronizer=zarr.ThreadSynchronizer(),
                dtype=dtype
            )
            dsfactors = [int(i)**mip_lvl for i in mip_dsfactor]
            mip_ds[mip_lvl] = ds_lvl
            scales.append(dsfactors)
        
        nblocks = [int(numpy.ceil(joined_shapes[k]/block_size[k])) for k in range(3)]
        print(str(nblocks) + " number of chunks per axis")
        print(str(g[0].nchunks) + " chunk number sanity check")

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as e:
            futs = []
            mips = []
            for miparr in iterate_mip_levels_from_dataset(
                    dataset, max_mip, max_mip, nblocks, block_size, mip_dsfactor,
                    num_slice=numchunks,chunknum=chunknum,
                    lvl_to_mip_kwargs=lvl_to_mip_kwargs,
                    interleaved_channels=interleaved_channels,
                    channel=channel, deskew_kwargs=deskew_kwargs):
                mips.append(miparr)
                if miparr.lvl == max_mip:
                    futs.append(e.submit(
                        write_mips, mip_ds, mips))
                    mips = []
            # for miparr in iterate_mip_levels_from_dataset(
            #         dataset, max_mip, max_mip, nblocks, block_size, mip_dsfactor,
            #         lvl_to_mip_kwargs=lvl_to_mip_kwargs,
            #         interleaved_channels=interleaved_channels,
            #         channel=channel, deskew_kwargs=deskew_kwargs):
            #     futs.append(e.submit(
            #         dswrite_block, mip_ds[miparr.lvl],
            #         miparr.start, miparr.end, miparr.array))
            for fut in concurrent.futures.as_completed(futs):
                _ = fut.result()
    print("conversion complete, closing file")
    #f.close()


def ims_to_ngff_group(ims_fn, output, *args, **kwargs):
    """convert directory of natsort-consecutive multitiffs to an n5 or zarr pyramid

    Parameters
    ----------
    tiffdir : str
        directory of consecutive multitiffs to convert
    """

    if output == 'zarr':
        print('converting to zarr')
        return write_ims_to_zarr(ims_fn, *args, **kwargs)
    else:
        print('unknown output format: ' + output)



class IMSToNGFFParameters(NGFFGroupGenerationParameters):
    input_file = argschema.fields.Str(required=True)
    interleaved_channels = argschema.fields.Int(required=False, default=1)
    channel = argschema.fields.Int(required=False, default=0)
    num_chunks = argschema.fields.Int(required=False, default=0)
    chunk_num = argschema.fields.Int(required=False, default=-1)


class IMSToZarrInputParameters(argschema.ArgSchema,
                                    IMSToNGFFParameters):
    chunk_size = argschema.fields.Tuple((
        argschema.fields.Int(),
        argschema.fields.Int(),
        argschema.fields.Int(),
        argschema.fields.Int(),
        argschema.fields.Int()), required=False, default=(1, 1, 64, 64, 64))



class IMSToZarr(argschema.ArgSchemaParser):
    default_schema = IMSToZarrInputParameters

    def run(self):
        deskew_options = (self.args["deskew_options"]
                          if "deskew_options" in self.args else {})
        ims_to_ngff_group(
            self.args["input_file"], self.args["output_format"],
            self.args["output_file"], self.args["group_names"],
            self.args["group_attributes"],
            self.args["max_mip"],
            self.args["mip_dsfactor"],
            self.args["chunk_size"],
            concurrency=self.args["concurrency"],
            compression=self.args["compression"],
            #lvl_to_mip_kwargs=self.args["lvl_to_mip_kwargs"],
            numchunks = self.args["num_chunks"],
            chunknum = self.args["chunk_num"],
            deskew_options=deskew_options)



if __name__ == "__main__":
    mod = IMSToZarr()
    mod.run()
