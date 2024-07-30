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
from acpreprocessing.stitching_modules.convert_to_n5.tiff_to_ngff import iterate_mip_levels_from_dataset,mip_level_shape,dswrite_chunk,omezarr_attrs,NGFFGroupGenerationParameters,TiffToNGFFValueError


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
        block_size = chunk_size[2]
        slice_length = block_size
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

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as e:
            futs = []
            for miparr in iterate_mip_levels_from_dataset(
                    dataset, max_mip, block_size, slice_length, mip_dsfactor,
                    lvl_to_mip_kwargs=lvl_to_mip_kwargs,
                    interleaved_channels=interleaved_channels,
                    channel=channel, deskew_kwargs=deskew_kwargs):
                futs.append(e.submit(
                    dswrite_chunk, mip_ds[miparr.lvl],
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
