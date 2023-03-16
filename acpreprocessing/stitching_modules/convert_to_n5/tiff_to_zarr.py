#!/usr/bin/env python

import argschema
from schemas import TiffDirToNGFFParameters, DeskewOptions
from convert_tiffs import tiffdir_to_ngff_group


class TiffDirToZarrInputParameters(argschema.ArgSchema,
                                   TiffDirToNGFFParameters,
                                   DeskewOptions):
    out_zarr = argschema.fields.Str(required=True)
    chunk_size = argschema.fields.Tuple((
        argschema.fields.Int(),
        argschema.fields.Int(),
        argschema.fields.Int(),
        argschema.fields.Int(),
        argschema.fields.Int()), required=False, default=(1, 1, 64, 64, 64))


class TiffDirToZarr(argschema.ArgSchemaParser):
    default_schema = TiffDirToZarrInputParameters
    deskew_options = DeskewOptions

    def run(self):
        deskew = DeskewOptions.dump(
            DeskewOptions.load(self.args, unknown="EXCLUDE"))
        tiffdir_to_ngff_group(
            self.args["input_dir"], "zarr",
            self.args["out_zarr"], self.args["group_names"],
            self.args["group_attributes"],
            self.args["max_mip"],
            self.args["mip_dsfactor"],
            self.args["chunk_size"],
            concurrency=self.args["concurrency"],
            compression=self.args["compression"],
            # lvl_to_mip_kwargs=self.args["lvl_to_mip_kwargs"],
            # FIXME not sure why this dict errors
            lvl_to_mip_kwargs={},
            deskew_options=deskew)


if __name__ == "__main__":
    mod = TiffDirToZarr()
    mod.run()
