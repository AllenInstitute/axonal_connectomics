#!/usr/bin/env python
"""write n5 format directory given an axonal connectomics
tiff style acquisition directory
"""
import concurrent.futures
import json
import pathlib
import shutil

import argschema
from acpreprocessing.stitching_modules.convert_to_n5.tiff_to_ngff import tiffdir_to_ngff_group, NGFFGenerationParameters


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


def get_position_names_from_rootdir(
        root_dir, stripjson_bn="hh.log",
        stripjson_key="stripdirs"):
    root_path = pathlib.Path(root_dir)
    stripjson_path = root_path / stripjson_bn
    with stripjson_path.open() as f:
        stripjson_md = json.load(f)
    return stripjson_md[stripjson_key]


def get_pixel_resolution_from_rootdir(
        root_dir, md_bn="acqinfo_metadata.json"):
    root_path = pathlib.Path(root_dir)
    md_path = root_path / md_bn
    with md_path.open() as f:
        md = json.load(f)
    xy = md["settings"]["pixel_spacing_um"]
    z = md["positions"][0]["x_step_um"]
    return [xy, xy, z]


def get_strip_positions_from_rootdir(
        root_dir, md_bn="acqinfo_metadata.json"):
    root_path = pathlib.Path(root_dir)
    md_path = root_path / md_bn
    with md_path.open() as f:
        md = json.load(f)
    # FIXME how to do proper version comparison?
    if md["version"] == "0.0.3":
        return [(p["z_start_um"], p["y_start_um"], p["x_start_um"]) for p in md["positions"]]
    else:
        return [(0, p["y_start_um"], p["x_start_um"]) for p in md["positions"]]


def get_number_interleaved_channels_from_rootdir(
        root_dir, md_bn="acqinfo_metadata.json"):
    root_path = pathlib.Path(root_dir)
    md_path = root_path / md_bn
    with md_path.open() as f:
        md = json.load(f)

    interleaved_channels = md.get("channels", 1)
    return interleaved_channels


def acquisition_to_ngff(acquisition_dir, output, out_dir, concurrency=5,
                        ngff_generation_kwargs=None, copy_top_level_files=True):
    """
    """
    ngff_generation_kwargs = (
        {} if ngff_generation_kwargs is None
        else ngff_generation_kwargs)

    acquisition_path = pathlib.Path(acquisition_dir)
    out_path = pathlib.Path(out_dir)
    if output == 'zarr':
        output_dir = str(out_path / f"{out_path.name}.zarr")
    else:
        output_dir = str(out_path / f"{out_path.name}.n5")

    interleaved_channels = get_number_interleaved_channels_from_rootdir(
        acquisition_path)
    positionList = get_strip_positions_from_rootdir(acquisition_path)

    try:
        setup_group_attributes = [{
            "pixelResolution": {
                "dimensions": get_pixel_resolution_from_rootdir(
                    acquisition_path),
                "unit": "um"
            },
            "position": p
        } for p in positionList]
    except (KeyError, FileNotFoundError):
        setup_group_attributes = {}

    for channel_idx in range(interleaved_channels):
        channel_group_attributes = {}
        with concurrent.futures.ProcessPoolExecutor(max_workers=concurrency) as e:
            futs = []
            for i, pospath in enumerate(
                    yield_position_paths_from_rootdir(
                        acquisition_path)):
                # pos_group = pospath.name
                # below is more like legacy structure
                # out_n5_dir = str(out_path / f"{pos_group}.n5")
                if output == 'zarr':
                    group_names = [pospath.name]
                    group_attributes = [setup_group_attributes[i]]
                else:
                    group_names = [
                        f"channel{channel_idx}", f"setup{i}", "timepoint0"]
                    group_attributes = [channel_group_attributes,
                                        setup_group_attributes[i]]

                futs.append(e.submit(
                    tiffdir_to_ngff_group,
                    str(pospath), output, output_dir, group_names,
                    group_attributes=group_attributes,
                    interleaved_channels=interleaved_channels,
                    channel=channel_idx,
                    **ngff_generation_kwargs
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


class AcquisitionDirToNGFFParameters(
        argschema.ArgSchema, NGFFGenerationParameters):
    input_dir = argschema.fields.Str(required=True)
    output_format = argschema.fields.Str(required=True)
    output_dir = argschema.fields.Str(required=True)
    copy_top_level_files = argschema.fields.Bool(required=False, default=True)
    position_concurrency = argschema.fields.Int(required=False, default=5)


class AcquisitionDirToNGFF(argschema.ArgSchemaParser):
    default_schema = AcquisitionDirToNGFFParameters

    def _get_ngff_kwargs(self):
        ngff_keys = {
            "max_mip", "concurrency", "compression",
            "lvl_to_mip_kwargs", "chunk_size", "mip_dsfactor",
            "deskew_method", "deskew_stride", "deskew_flip", "deskew_crop"}
        return {k: self.args[k] for k in (ngff_keys & self.args.keys())}

    def run(self):
        ngff_kwargs = self._get_ngff_kwargs()
        acquisition_to_ngff(
            self.args["input_dir"], self.args["output_format"], self.args["output_dir"],
            concurrency=self.args["position_concurrency"],
            ngff_generation_kwargs=ngff_kwargs,
            copy_top_level_files=self.args["copy_top_level_files"]
        )


if __name__ == "__main__":
    mod = AcquisitionDirToNGFF()
    mod.run()
