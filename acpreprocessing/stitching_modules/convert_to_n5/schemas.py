#!/usr/bin/env python

import argschema


class DownsampleOptions(argschema.schemas.DefaultSchema):
    block_divs = argschema.fields.List(
        argschema.fields.Int, required=False, allow_none=True)
    n_threads = argschema.fields.Int(required=False, allow_none=True)


class DeskewOptions(argschema.schemas.DefaultSchema):
    deskew_method = argschema.fields.Str(required=False, default='')
    deskew_stride = argschema.fields.Int(required=False, default=None)
    deskew_flip = argschema.fields.Bool(required=False, default=True)
    deskew_crop = argschema.fields.Float(required=False, default=0.5)


class NGFFGenerationParameters(argschema.schemas.DefaultSchema):
    max_mip = argschema.fields.Int(required=False, default=0)
    concurrency = argschema.fields.Int(required=False, default=10)
    compression = argschema.fields.Str(required=False, default="raw")
    lvl_to_mip_kwargs = argschema.fields.Dict(
        keys=argschema.fields.Int(),
        values=argschema.fields.Nested(DownsampleOptions))

    # FIXME argschema supports lists and tuples,
    #   but has some version differences

    mip_dsfactor = argschema.fields.Tuple((
        argschema.fields.Int(),
        argschema.fields.Int(),
        argschema.fields.Int()), required=False, default=(2, 2, 2))


class NGFFGroupGenerationParameters(NGFFGenerationParameters):
    group_names = argschema.fields.List(
        argschema.fields.Str, required=True)
    group_attributes = argschema.fields.List(
        argschema.fields.Dict(required=False, default={}), default=[],
        required=False)


class TiffDirToNGFFParameters(NGFFGroupGenerationParameters):
    input_dir = argschema.fields.InputDir(required=True)
    interleaved_channels = argschema.fields.Int(required=False, default=1)
    channel = argschema.fields.Int(required=False, default=0)
    attributes_json = argschema.fields.Str(required=False, default='')
