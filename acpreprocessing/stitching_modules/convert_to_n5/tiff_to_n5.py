#!/usr/bin/env python

import ast
import concurrent.futures
import dataclasses
import itertools
import math
import pathlib
from natsort import natsorted

import imageio
import numpy

import z5py
import argschema

import os
import shutil

import acpreprocessing.utils.convert


def iterate_chunks(it, slice_length):
    it = iter(it)
    chunk = tuple(itertools.islice(it, slice_length))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, slice_length))


def iter_arrays(r):
    for i, p in enumerate(r._tf.pages):
        arr = p.asarray()
        if arr is not None:
            yield arr
        elif i == r.get_length() - 1:
            continue
        else:
            raise ValueError

def iterate_2d_arrays_from_mimgfns(mimgfns):
    for mimgfn in mimgfns:
        print(mimgfn)
        with imageio.get_reader(mimgfn, mode="I") as r:
            #numpy.array([i for i in iter_arrays(r)])
            yield from iter_arrays(r)

def iterate_numpy_chunks_from_mimgfns(mimgfns, slice_length=None, pad=True):
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


def mimg_shape_from_fn(mimg_fn):
    with imageio.get_reader(mimg_fn, mode="I") as r:
        s = (r.get_length(), *r.get_data(0).shape)
    return s


def joined_mimg_shape_from_fns(mimg_fns):
    shapes = [mimg_shape_from_fn(fn) for fn in mimg_fns]
    return (
        sum([s[0] for s in shapes]),
        shapes[0][1],
        shapes[0][2]
    )


def mip_level_shape(lvl, lvl_0_shape):
    return tuple([*map(lambda x: math.ceil(x / (2**lvl)), lvl_0_shape)])


def dswrite_chunk(ds, start, end, arr):
    ds[start:end, :, :] = arr[:(end - start), :, :]


@dataclasses.dataclass
class MIPArray:
    lvl: int
    array: numpy.ndarray
    start: int
    end: int


def iterate_mip_levels_from_mimgfns(
        mimgfns, lvl, block_size,
        downsample_factor, downsample_method=None):
    """
    iterator that uses temporary arrays to generate downsample MIP levels as
      higher resoulution data is read in.  Currently pretty use-case specific.
    """
    start_index = 0
    if lvl > 0:
        num_chunks = downsample_factor[0]

        i = 0
        for ma in iterate_mip_levels_from_mimgfns(
                mimgfns, lvl-1, block_size,
                downsample_factor, downsample_method):
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

            # yield the full chunk when completed
            if i == num_chunks - 1:
                temp_arr = (
                    acpreprocessing.utils.convert.downsample_stack_volume(
                        temp_lminus1_arr, downsample_factor,
                        dtype=chunk.dtype, method=downsample_method))
                end_index = start_index + temp_arr.shape[0]
                yield MIPArray(lvl, temp_arr, start_index, end_index)
                start_index += temp_arr.shape[0]
                i = 0
            else:
                i += 1
        # add any leftovers
        if i > 0:
            # FIXME this might be assuming 2x downsample
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
        mimgfns, out_n5, ds_name, chunk_size=(64, 64, 64),
        max_mip=0, mip_dsfactor=(2, 2, 2), concurrency=1,
        compression="raw"):
    # TODO different chunk sizes for different levels
    joined_shapes = joined_mimg_shape_from_fns(mimgfns)
    # TODO also get dtype from mimg
    dtype = "uint16"
    slice_length = chunk_size[0]

    with z5py.File(out_n5) as f:
        mip_ds = {}
        for mip_lvl in range(max_mip+1):
            g_lvl = f.create_group(f"s{mip_lvl}")
            ds_lvl = g_lvl.create_dataset(
                ds_name,
                chunks=chunk_size,
                shape=mip_level_shape(mip_lvl, joined_shapes),
                compression=compression,
                dtype=dtype
            )
            mip_ds[mip_lvl] = ds_lvl

        with concurrent.futures.ThreadPoolExecutor(
                max_workers=concurrency) as e:
            futs = []
            for miparr in iterate_mip_levels_from_mimgfns(
                    mimgfns, max_mip, slice_length, mip_dsfactor):
                futs.append(e.submit(
                    dswrite_chunk,
                    mip_ds[miparr.lvl], miparr.start, miparr.end,
                    miparr.array))
            for fut in concurrent.futures.as_completed(futs):
                _ = fut.result()


class TiffDirToN5InputParameters(argschema.ArgSchema):
    input_dir = argschema.fields.InputDir(required=True)
    out_n5 = argschema.fields.Str(required=True)
    ds_name = argschema.fields.Str(required=True)
    max_mip = argschema.fields.Int(required=False, default=0)
    concurrency = argschema.fields.Int(required=False, default=10)
    compression = argschema.fields.Str(required=False, default="raw")

    # FIXME argschema supports lists and tuples,
    #   but has some version differences
    chunk_size = argschema.fields.Str(required=False, default="64,64,64")
    mip_dsfactor = argschema.fields.Str(required=False, default="2,2,2")


class TiffDirToN5(argschema.ArgSchemaParser):
    default_schema = TiffDirToN5InputParameters

    def run(self):
        
        files = [*sorted(pathlib.Path(self.args["input_dir"]).iterdir())]
        mimgfns = natsorted([str(p) for p in files if p.name.endswith('.tif')])
        
        # FIXME argschema can and should do this
        chunk_size = ast.literal_eval(self.args["chunk_size"])
        mip_dsfactor = ast.literal_eval(self.args["mip_dsfactor"])

        write_mimgfns_to_n5(
            mimgfns, self.args["out_n5"], self.args["ds_name"],
            chunk_size, self.args["max_mip"], mip_dsfactor,
            self.args["concurrency"], self.args["compression"])

        #Fix dir structure
        multires = os.path.join(self.args["out_n5"],"multires"+self.args["ds_name"])

        fullres = os.path.join(self.args["out_n5"],"s0/"+self.args["ds_name"])
        shutil.move(fullres, self.args["out_n5"])

        try:
            os.rmdir(os.path.join(self.args["out_n5"],"s0"))
        except OSError as e:
            print("Error: %s : %s" % (dir_path, e.strerror))

        os.makedirs(multires)
        for mip_level in range(1,self.args["max_mip"]+1):
            source = os.path.join(self.args["out_n5"],os.path.join(f"s{mip_level}/",self.args["ds_name"]))
            shutil.move(source, os.path.join(multires,f"s{mip_level}"))
            try:
                os.rmdir(os.path.join(self.args["out_n5"],f"s{mip_level}"))
            except OSError as e:
                print("Error: %s : %s" % (dir_path, e.strerror))

if __name__ == "__main__":
    mod = TiffDirToN5()
    mod.run()

