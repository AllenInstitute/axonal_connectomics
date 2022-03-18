#!/usr/bin/env python
"""write n5 format directory given an axonal connectomics
tiff style acquisition directory
"""
import concurrent.futures
import json
import pathlib
import shutil

import argschema

from acpreprocessing.stitching_modules.convert_to_n5.tiff_to_n5 import (
    tiffdir_to_n5_group,
    N5GenerationParameters
)


def yield_position_paths_from_rootdir(
        root_dir, stripjson_bn="hh.log",
        stripjson_key="stripdirs"):
    root_path = pathlib.Path(root_dir)
    stripjson_path = root_path / stripjson_bn
    with stripjson_path.open() as f:
        stripjson_md = json.load(f)
    stripdirs_bns = stripjson_md[stripjson_key]
    for stripdir_bn in stripdirs_bns:
        yield root_path / stripdir_bn


def get_pixel_resolution_from_rootdir(
        root_dir, md_bn="acquisition_metadata.json"):
    root_path = pathlib.Path(root_dir)
    md_path = root_path / md_bn
    with md_path.open() as f:
        md = json.load(f)
    xy = md["settings"]["pixel_spacing_um"]
    z = md["positions"][1]["x_step_um"]
    return [xy, xy, z]


def acquisition_to_n5(acquisition_dir, out_dir, concurrency=5,
                      n5_generation_kwargs=None, copy_top_level_files=True):
    """
    """
    n5_generation_kwargs = (
        {} if n5_generation_kwargs is None
        else n5_generation_kwargs)

    acquisition_path = pathlib.Path(acquisition_dir)
    out_path = pathlib.Path(out_dir)
    out_n5_dir = str(out_path / f"{out_path.name}.n5")

    try:
        group_attributes = {
            "pixelResolution": {
                "dimensions": get_pixel_resolution_from_rootdir(
                    acquisition_path),
                "unit": "um"
            }
        }
    except (KeyError, FileNotFoundError):
        group_attributes = {}

    with concurrent.futures.ProcessPoolExecutor(max_workers=concurrency) as e:
        futs = []
        for i, pospath in enumerate(
                yield_position_paths_from_rootdir(
                    acquisition_path)):
            # pos_group = pospath.name
            # below is more like legacy structure
            # out_n5_dir = str(out_path / f"{pos_group}.n5")
            futs.append(e.submit(
                tiffdir_to_n5_group,
                str(pospath), out_n5_dir, [f"setup{i}", "timepoint0"],
                group_attributes=[group_attributes], **n5_generation_kwargs
            ))

        for fut in concurrent.futures.as_completed(futs):
            _ = fut.result()

    if copy_top_level_files:
        top_level_files_paths = [
            p for p in acquisition_path.iterdir()
            if p.is_file()]
        for tlf_path in top_level_files_paths:
            out_tlf_path = out_path / tlf_path.name
            shutil.copy(str(tlf_path), str(out_tlf_path))


class AcquisitionDirToN5DirParameters(
        argschema.ArgSchema, N5GenerationParameters):
    input_dir = argschema.fields.Str(required=True)
    output_dir = argschema.fields.Str(required=True)
    copy_top_level_files = argschema.fields.Bool(required=False, default=True)
    position_concurrency = argschema.fields.Int(required=False, default=5)


class AcquisitionDirToN5Dir(argschema.ArgSchemaParser):
    default_schema = AcquisitionDirToN5DirParameters

    def _get_n5_kwargs(self):
        n5_keys = {
            "max_mip", "concurrency", "compression",
            "lvl_to_mip_kwargs", "chunk_size", "mip_dsfactor"}
        return {k: self.args[k] for k in (n5_keys & self.args.keys())}

    def run(self):
        n5_kwargs = self._get_n5_kwargs()
        acquisition_to_n5(
            self.args["input_dir"], self.args["output_dir"],
            concurrency=self.args["position_concurrency"],
            n5_generation_kwargs=n5_kwargs,
            copy_top_level_files=self.args["copy_top_level_files"]
            )


if __name__ == "__main__":
    mod = AcquisitionDirToN5Dir()
    mod.run()
