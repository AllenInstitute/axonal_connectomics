#!/usr/bin/env python

import concurrent.futures
import dataclasses
import itertools
import math
import pathlib

import imageio
from natsort import natsorted
import numpy
import skimage

import z5py
import argschema

import acpreprocessing.utils.convert


def iterate_chunks(it, slice_length):
    """given an iterator, iterate over tuples of a
    given length from that iterator
    """
    it = iter(it)
    chunk = tuple(itertools.islice(it, slice_length))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, slice_length))


def iter_arrays(r):
    """iterate arrays from an imageio tiff reader.  Allows the last image
    of the array to be None, as is the case for data with 'dropped frames'.
    """
    for i, p in enumerate(r._tf.pages):
        arr = p.asarray()
        if arr is not None:
            yield arr
        elif i == r.get_length() - 1:
            continue
        else:
            raise ValueError


def iterate_2d_arrays_from_mimgfns(mimgfns):
    """iterate constituent arrays from an iterator of image filenames
    that can be opened as an imageio multi-image.
    """
    for mimgfn in mimgfns:
        print(mimgfn)
        with imageio.get_reader(mimgfn, mode="I") as r:
            yield from iter_arrays(r)


def iterate_numpy_chunks_from_mimgfns(mimgfns, slice_length=None, pad=True):
    """iterate over a contiguous iterator of imageio multi-image files as
    chunks of numpy arrays
    """
    array_gen = iterate_2d_arrays_from_mimgfns(mimgfns)
    for chunk in iterate_chunks(array_gen, slice_length):
        arr = numpy.array(chunk)
        if pad:
            if arr.shape[0] != slice_length:
                newarr = numpy.zeros((slice_length, *arr.shape[1:]),
                                     dtype=arr.dtype)
                newarr[:arr.shape[0], :, :] = arr[:, :, :]
                yield newarr
            else:
                yield arr
        else:
            yield arr


def mimg_shape_from_fn(mimg_fn, only_length_tup=False):
    """get the shape of an imageio multi-image file without
    reading it as a volume
    """
    with imageio.get_reader(mimg_fn, mode="I") as r:
        if only_length_tup:
            s = (r.get_length(),)
        else:
            s = (r.get_length(), *r.get_data(0).shape)
    return s


def joined_mimg_shape_from_fns(mimg_fns, concurrency=1):
    """concurrently read image shapes from tiff files representing a
    contiguous stack to get the shape of the combined stack.
    """
    # join shapes while reading concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as e:
        futs = [e.submit(mimg_shape_from_fn, fn) for fn in mimg_fns]
        shapes = [fut.result() for fut
                  in concurrent.futures.as_completed(futs)]
    return (
        sum([s[0] for s in shapes]),
        shapes[0][1],
        shapes[0][2]
    )


def array_downsample_chunks(arr, ds_factors, block_shape, **kwargs):
    """downsample a numpy array by inputting subvolumes
    to acpreprocessing.utils.convert.downsample_stack_volume
    """
    blocks = skimage.util.view_as_blocks(arr, block_shape)

    # construct output array and blocks
    output_shape = tuple(
        int(math.ceil(s / f)) for s, f in zip(arr.shape, ds_factors))
    output_block_shape = tuple(
        int(math.ceil(s / f)) for s, f in zip(block_shape, ds_factors))
    output_arr = numpy.empty(output_shape, dtype=arr.dtype)
    output_blocks = skimage.util.view_as_blocks(
        output_arr, output_block_shape)

    # process
    for idxs in numpy.ndindex(blocks.shape[:arr.ndim]):
        output_blocks[idxs] = (
            acpreprocessing.utils.convert.downsample_stack_volume(
                blocks[idxs], ds_factors, **kwargs))

    return output_arr


def array_downsample_threaded(arr, ds_factors, block_shape,
                              n_threads=3, **kwargs):
    """downsample a numpy array by concurrently inputting subvolumes
    to acpreprocessing.utils.convert.downsample_stack_volume
    """
    blocks = skimage.util.view_as_blocks(arr, block_shape)

    # construct output array and blocks
    output_shape = tuple(
        int(math.ceil(s / f)) for s, f in zip(arr.shape, ds_factors))
    output_block_shape = tuple(
        int(math.ceil(s / f)) for s, f in zip(block_shape, ds_factors))
    output_arr = numpy.empty(output_shape, dtype=arr.dtype)
    output_blocks = skimage.util.view_as_blocks(
        output_arr, output_block_shape)

    # process
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as e:
        fut_to_idxs = {}
        for idxs in numpy.ndindex(blocks.shape[:arr.ndim]):
            fut_to_idxs[e.submit(
                acpreprocessing.utils.convert.downsample_stack_volume,
                blocks[idxs], ds_factors, **kwargs)] = idxs

        for fut in concurrent.futures.as_completed(fut_to_idxs.keys()):
            idxs = fut_to_idxs[fut]
            output_blocks[idxs] = fut.result()
    return output_arr


class ArrayChunkingError(ValueError):
    """ValueError caused by incorrect array chunking"""


def downsample_array(arr, *args, n_threads=None, block_shape=None,
                     block_divs=None, allow_failover=True, **kwargs):
    """downsample array with optional subvolume chunking and threading
    """
    try:
        if block_shape is not None or block_divs is not None:
            if block_shape is None:
                block_shape = [a_s / d for a_s, d in zip(
                    arr.shape, block_divs)]
                if not all([s.is_integer() for s in block_shape]):
                    raise ArrayChunkingError(
                        "cannot divide array of shape {} by {}".format(
                            arr.shape, block_divs))
                block_shape = tuple(int(s) for s in block_shape)
            if n_threads is not None:
                return array_downsample_threaded(
                    arr, *args, n_threads=n_threads,
                    block_shape=block_shape, **kwargs)
            return array_downsample_chunks(
                arr, *args, block_shape=block_shape, **kwargs)
    except ArrayChunkingError:
        if not allow_failover:
            raise
    return acpreprocessing.utils.convert.downsample_stack_volume(
        arr, *args, **kwargs)


def mip_level_shape(lvl, lvl_0_shape):
    """get the expected shape of a MIP level downsample for a given
    full resolution shape
    """
    return tuple([*map(lambda x: math.ceil(x / (2**lvl)), lvl_0_shape)])


def dswrite_chunk(ds, start, end, arr, silent_overflow=True):
    if end >= ds.shape[0] and silent_overflow:
        end = ds.shape[0]
    ds[start:end, :, :] = arr[:(end - start), :, :]


@dataclasses.dataclass
class MIPArray:
    lvl: int
    array: numpy.ndarray
    start: int
    end: int


def iterate_mip_levels_from_mimgfns(
        mimgfns, lvl, block_size, downsample_factor,
        downsample_method=None, lvl_to_mip_kwargs=None):
    """recursively generate MIPmap levels from an iterator of multi-image files
    """
    lvl_to_mip_kwargs = ({} if lvl_to_mip_kwargs is None
                         else lvl_to_mip_kwargs)
    mip_kwargs = lvl_to_mip_kwargs.get(lvl, {})
    start_index = 0
    if lvl > 0:
        num_chunks = downsample_factor[0]

        i = 0
        for ma in iterate_mip_levels_from_mimgfns(
                mimgfns, lvl-1, block_size,
                downsample_factor, downsample_method,
                lvl_to_mip_kwargs):
            chunk = ma.array
            # throw array up for further processing
            yield ma
            # ignore if not parent resolution
            if ma.lvl != lvl-1:
                continue

            try:
                temp_lminus1_arr
            except NameError:
                temp_lminus1_arr = numpy.empty(
                    (num_chunks*chunk.shape[0], *chunk.shape[1:]),
                    dtype=chunk.dtype)

            # fill in temporary block according to index
            chunk_size = chunk.shape[0]
            block_offset = i*block_size

            temp_lminus1_arr[
                block_offset:block_offset+chunk_size, :, :] = chunk[:, :, :]

            # copy op only for uneven final chunk
            if chunk_size != block_size:
                temp_lminus1_arr = temp_lminus1_arr[
                    :block_offset+chunk_size, :, :]

            if i == num_chunks - 1:
                temp_arr = (
                    downsample_array(
                        temp_lminus1_arr, downsample_factor, dtype=chunk.dtype,
                        method=downsample_method, **mip_kwargs))
                end_index = start_index + temp_arr.shape[0]
                yield MIPArray(lvl, temp_arr, start_index, end_index)
                start_index += temp_arr.shape[0]
                i = 0
            else:
                i += 1
        # add any leftovers
        if i > 0:
            temp_arr = acpreprocessing.utils.convert.downsample_stack_volume(
                temp_lminus1_arr, downsample_factor, dtype=chunk.dtype,
                method=downsample_method)
            end_index = start_index + temp_arr.shape[0]
            yield MIPArray(lvl, temp_arr, start_index, end_index)
    else:
        # get level 0 chunks
        for chunk in iterate_numpy_chunks_from_mimgfns(
                mimgfns, block_size, pad=False):
            end_index = start_index + chunk.shape[0]
            yield MIPArray(lvl, chunk, start_index, end_index)
            start_index += chunk.shape[0]


def write_mimgfns_to_n5(
        mimgfns, output_n5, group_names, group_attributes=None, max_mip=0,
        mip_dsfactor=(2, 2, 2), chunk_size=(64, 64, 64),
        concurrency=10, slice_concurrency=1,
        compression="raw", dtype="uint16", lvl_to_mip_kwargs=None):
    """write a stack represented by an iterator of multi-image files as an n5
    volume
    """
    group_attributes = ([] if group_attributes is None else group_attributes)

    joined_shapes = joined_mimg_shape_from_fns(
        mimgfns, concurrency=concurrency)
    # TODO also get dtype from mimg

    slice_length = chunk_size[0]

    workers = concurrency // slice_concurrency

    with z5py.File(output_n5) as f:
        mip_ds = {}
        # create groups with custom attributes
        group_objs = []
        for i, group_name in enumerate(group_names):
            try:
                g = group_objs[-1].create_group(f"{group_name}")
            except IndexError:
                g = f.create_group(f"{group_name}")
            group_objs.append(g)
            try:
                attributes = group_attributes[i]
            except IndexError:
                continue
            for k, v in attributes.items():
                g.attrs[k] = v
        scales = []
        for mip_lvl in range(max_mip + 1):
            ds_lvl = g.create_dataset(
                f"s{mip_lvl}",
                chunks=chunk_size,
                shape=mip_level_shape(mip_lvl, joined_shapes),
                compression=compression,
                dtype=dtype,
                n_threads=slice_concurrency)
            dsfactors = [int(i)**mip_lvl for i in mip_dsfactor]
            ds_lvl.attrs["downsamplingFactors"] = dsfactors
            mip_ds[mip_lvl] = ds_lvl
            scales.append(dsfactors)
        g.attrs["scales"] = scales
        group_objs[0].attrs["downsamplingFactors"] = scales
        group_objs[0].attrs["dataType"] = dtype

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as e:
            futs = []
            for miparr in iterate_mip_levels_from_mimgfns(
                    mimgfns, max_mip, slice_length, mip_dsfactor,
                    lvl_to_mip_kwargs=lvl_to_mip_kwargs):
                futs.append(e.submit(
                    dswrite_chunk, mip_ds[miparr.lvl],
                    miparr.start, miparr.end, miparr.array))
            for fut in concurrent.futures.as_completed(futs):
                _ = fut.result()


def tiffdir_to_n5_group(tiffdir, *args, **kwargs):
    mimgfns = [str(p) for p in natsorted(
                   pathlib.Path(tiffdir).iterdir(), key=lambda x:str(x))
               if p.is_file()]
    return write_mimgfns_to_n5(mimgfns, *args, **kwargs)


class DownsampleOptions(argschema.schemas.DefaultSchema):
    block_divs = argschema.fields.List(
        argschema.fields.Int, required=False, allow_none=True)
    n_threads = argschema.fields.Int(required=False, allow_none=True)


class N5GenerationParameters(argschema.schemas.DefaultSchema):
    max_mip = argschema.fields.Int(required=False, default=0)
    concurrency = argschema.fields.Int(required=False, default=10)
    compression = argschema.fields.Str(required=False, default="raw")
    lvl_to_mip_kwargs = argschema.fields.Dict(
        keys=argschema.fields.Int(),
        values=argschema.fields.Nested(DownsampleOptions))

    # FIXME argschema supports lists and tuples,
    #   but has some version differences
    chunk_size = argschema.fields.Tuple((
        argschema.fields.Int(),
        argschema.fields.Int(),
        argschema.fields.Int()), required=False, default=(64, 64, 64))
    mip_dsfactor = argschema.fields.Tuple((
        argschema.fields.Int(),
        argschema.fields.Int(),
        argschema.fields.Int()), required=False, default=(2, 2, 2))


class N5GroupGenerationParameters(N5GenerationParameters):
    group_names = argschema.fields.List(
        argschema.fields.Str, required=True)
    group_attributes = argschema.fields.List(
        argschema.fields.Dict(required=False, default={}), default=[],
        required=False)


class TiffDirToN5InputParameters(argschema.ArgSchema,
                                 N5GroupGenerationParameters):
    input_dir = argschema.fields.InputDir(required=True)
    out_n5 = argschema.fields.Str(required=True)


class TiffDirToN5(argschema.ArgSchemaParser):
    default_schema = TiffDirToN5InputParameters

    def run(self):
        tiffdir_to_n5_group(
            self.args["input_dir"], self.args["out_n5"],
            self.args["group_names"], self.args["group_attributes"],
            self.args["max_mip"],
            self.args["mip_dsfactor"],
            self.args["chunk_size"],
            concurrency=self.args["concurrency"],
            compression=self.args["compression"],
            lvl_to_mip_kwargs=self.args["lvl_to_mip_kwargs"])


if __name__ == "__main__":
    mod = TiffDirToN5()
    mod.run()
