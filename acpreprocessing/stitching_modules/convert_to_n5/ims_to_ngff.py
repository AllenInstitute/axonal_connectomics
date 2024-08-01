#!/usr/bin/env python

import concurrent.futures
import dataclasses
#import itertools
import math
#import pathlib

import imageio
#from natsort import natsorted
import numpy
import skimage

import h5py
import hdf5plugin
import zarr
from numcodecs import Blosc
import argschema

# import acpreprocessing.utils.convert
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


# def iterate_numpy_chunks_from_dataset(
#         dataset, block_size=None, pad=True, *args, **kwargs):
#     """iterate over a contiguous hdf5 daataset as chunks of numpy arrays

#     Parameters
#     ----------
#     dataset : hdf5 dataset
#         imageio-compatible name inputs to be opened as multi-images
#     slice_length : int
#         number of 2d arrays from mimgfns included per chunk
#     pad : bool, optional
#         whether to extend final returned chunks with zeros

#     Yields
#     ------
#     arr : numpy.ndarray
#         3D numpy array representing a consecutive chunk of 2D arrays
#     """
#     for chunk in iterate_chunks(dataset, block_size):#,*args,**kwargs):
#         arr = numpy.asarray(chunk)
#         if pad:
#             if any([arr.shape[k] != block_size[k] for k in range(3)]):
#                 newarr = numpy.zeros(block_size,
#                                       dtype=arr.dtype)
#                 newarr[:arr.shape[0], :arr.shape[1], :arr.shape[2]] = arr[:, :, :]
#                 yield newarr
#             else:
#                 yield arr
#         else:
#             yield arr


def iterate_numpy_blocks_from_dataset(
        dataset, maxlvl, nblocks, block_size=None, pad=True, deskew_kwargs={}, *args, **kwargs):
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
    for i in range(numpy.prod(nblocks)):#,*args,**kwargs):
        chunk_tuple = numpy.unravel_index(i,tuple(nblocks))
        #print(str(chunk_tuple))
        block_start = [chunk_tuple[k]*block_size[k] for k in range(3)]
        block_end = [block_start[k] + block_size[k] for k in range(3)]
        if deskew_kwargs:
            if deskew_kwargs["deskew_method"] == "ps":
                pass
                #arr = psd.get_deskewed_block(dataset,xi,yi,zi,**deskew_kwargs)
        else:
            arr = dataset[block_start[0]:block_end[0],block_start[1]:block_end[1],block_start[2]:block_end[2]]
        if pad:
            if any([arr.shape[k] != block_size[k] for k in range(3)]):
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
        downsample_method=None, lvl_to_mip_kwargs=None,
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
    if lvl > 0:
        for ma in iterate_mip_levels_from_dataset(
                dataset, lvl-1, maxlvl, nblocks, block_size,
                downsample_factor, downsample_method,
                lvl_to_mip_kwargs, interleaved_channels=interleaved_channels,
                channel=channel, deskew_kwargs=deskew_kwargs):
            chunk = ma.array
            # throw array up for further processing
            yield ma
            # ignore if not parent resolution
            if ma.lvl != lvl-1:
                continue
            
            temp_arr = (
                downsample_array(
                    chunk, downsample_factor, dtype=chunk.dtype,
                    method=downsample_method, **mip_kwargs))
            chunk_start = tuple([int(ma.start[k]/(downsample_factor[k]**lvl)) for k in range(3)])
            chunk_end = tuple([chunk_start[k] + temp_arr.shape[k] for k in range(3)])
            yield MIPArray(lvl, temp_arr, chunk_start, chunk_end)
    else:
        # get level 0 chunks
        # block_size is the number of slices to read from tiffs
        for block in iterate_numpy_blocks_from_dataset(
                dataset, maxlvl, nblocks, block_size=block_size, pad=False,
                channel=channel):
            # deskew level 0 chunk
            # if deskew_kwargs:
            #     chunk = numpy.transpose(psd.deskew_block(
            #         chunk, chunk_index, **deskew_kwargs), (2, 1, 0))
            block_tuple = numpy.unravel_index(block_index,nblocks)
            block_start = tuple([block_tuple[k]*block_size[k] for k in range(3)])
            block_end = tuple([block_start[k] + block.shape[k] for k in range(3)])
            yield MIPArray(lvl, block, block_start, block_end)
            block_index += 1


def write_ims_to_zarr(
        ims_fn, output_n5, group_names, group_attributes=None, max_mip=0,
        mip_dsfactor=(2, 2, 2), chunk_size=(1, 1, 64, 64, 64),
        concurrency=10, slice_concurrency=1,
        compression="raw", dtype="uint16", lvl_to_mip_kwargs=None,
        interleaved_channels=1, channel=0, deskew_options=None, **kwargs):
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
    
    f = h5py.File(ims_fn, 'r')
    dataset = f['DataSet']['ResolutionLevel 0']['TimePoint 0']['Channel 0']['Data']
    
    joined_shapes = dataset.shape
    print("ims_to_ngff dataset shape:" + str(joined_shapes))

    if deskew_options and deskew_options["deskew_method"] == "ps":
        block_size = chunk_size[2]
        slice_length = int(chunk_size[2]/deskew_options['deskew_stride'])
        deskew_kwargs = psd.psdeskew_kwargs(skew_dims_zyx=(slice_length, joined_shapes[1], joined_shapes[2]),
                                            **deskew_options
                                            )
        joined_shapes = psd.reshape_joined_shapes(
            joined_shapes, deskew_options['deskew_stride'], **deskew_kwargs)
        print("deskewed shape:" + str(joined_shapes))
    else:
        block_size = chunk_size[2:]
        deskew_kwargs = {}

    # TODO DESKEW: does this generally work regardless of skew?

    workers = concurrency // slice_concurrency

    zstore = zarr.DirectoryStore(output_n5, dimension_separator='/')
    with zarr.open(zstore, mode='a') as f:
        mip_ds = {}
        # create groups with attributes according to omezarr spec
        if len(group_names) == 1:
            group_name = group_names[0]
            try:
                g = f.create_group(f"{group_name}")
            except KeyError:
                g = f[f"{group_name}"]
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
                dtype=dtype
            )
            dsfactors = [int(i)**mip_lvl for i in mip_dsfactor]
            mip_ds[mip_lvl] = ds_lvl
            scales.append(dsfactors)
        
        nblocks = [int(numpy.ceil(joined_shapes[k]/block_size[k])) for k in range(3)]
        print(str(nblocks) + "number of chunks per axis")
        print(str(g[0].nchunks) + " chunk number sanity check")

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as e:
            futs = []
            for miparr in iterate_mip_levels_from_dataset(
                    dataset, max_mip, max_mip, nblocks, block_size, mip_dsfactor,
                    lvl_to_mip_kwargs=lvl_to_mip_kwargs,
                    interleaved_channels=interleaved_channels,
                    channel=channel, deskew_kwargs=deskew_kwargs):
                futs.append(e.submit(
                    dswrite_block, mip_ds[miparr.lvl],
                    miparr.start, miparr.end, miparr.array))
            for fut in concurrent.futures.as_completed(futs):
                _ = fut.result()
    print("conversion complete, closing file")
    f.close()


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
            deskew_options=deskew_options)



if __name__ == "__main__":
    mod = IMSToZarr()
    mod.run()
