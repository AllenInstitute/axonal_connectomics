#!/usr/bin/env python

import concurrent.futures
import pathlib
import shutil

import zarr

import argschema


def copy_dir_top_files(src, dst):
    src = pathlib.Path(src)
    dst = pathlib.Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for i in src.iterdir():
        if i.is_file():
            shutil.copy(i, dst / i.name)


def yield_in_out_filepaths(in_dir_path, out_dir_path):
    in_dir_path = pathlib.Path(in_dir_path)
    out_dir_path = pathlib.Path(out_dir_path)
    for in_item in in_dir_path.iterdir():
        out_item = out_dir_path / in_item.name
        if in_item.is_file():
            yield (in_item, out_item)
        if in_item.is_dir():
            yield from yield_in_out_filepaths(in_item, out_item)


def copy_path_mkdirs(in_path, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(in_path, out_path)
        

def fast_copy_tree(src, dst, concurrency=5):
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as e:
        futs = []
        for in_path, out_path in yield_in_out_filepaths(src, dst):
            futs.append(e.submit(copy_path_mkdirs, in_path, out_path))

        for fut in concurrent.futures.as_completed(futs):
            _ = fut.result()


def copy_zarr_group(zarr_group, new_base, only_top_files=True, concurrency=20):
    input_path = pathlib.Path(zarr_group.chunk_store.dir_path()) / zarr_group.path
    output_path = pathlib.Path(new_base) / zarr_group.path

    if only_top_files:
        copy_dir_top_files(input_path, output_path)
    else:
        fast_copy_tree(input_path, output_path, concurrency=concurrency)


def copy_zarr_levels_from_base(input_zarr_base, output_zarr_base,
                               levels_to_copy=("1",), concurrency=5):
    input_zarr_base = pathlib.Path(input_zarr_base)
    output_zarr_base = pathlib.Path(output_zarr_base)
    levels_to_copy = set(levels_to_copy)

    acq_group = zarr.open(input_zarr_base, mode="r")

    copy_zarr_group(acq_group, output_zarr_base,
                    only_top_files=True, concurrency=concurrency)
    for position_key, position_group in acq_group.items():
        copy_zarr_group(position_group, output_zarr_base,
                        only_top_files=True, concurrency=concurrency)
        for level_key, level_ds in position_group.items():
            if level_key in levels_to_copy:
                copy_zarr_group(
                    level_ds, output_zarr_base,
                    only_top_files=False, concurrency=concurrency)
            else:
                copy_zarr_group(
                    level_ds, output_zarr_base,
                    only_top_files=True, concurrency=concurrency)


class ZarrMIPTransferParameters(argschema.ArgSchema):
    levels_to_copy = argschema.fields.Tuple(
        argschema.fields.Int, default=(1,), required=False)
    input_zarr = argschema.fields.InputPath(required=True)
    output_dir = argschema.fields.OutputPath(required=True)
    concurrency = argschema.fields.Int(default=5, required=False)


class ZarrMIPTransferModule(argschema.ArgSchemaParser):
    default_schema = ZarrMIPTransferParameters

    def run(self):
        input_zarr = pathlib.Path(self.args["input_zarr"])
        output_zarr = pathlib.Path(self.args["output_dir"]) / input_zarr.name
        levels_to_copy = {str(lvl) for lvl in self.args["levels_to_copy"]}
        copy_zarr_levels_from_base(
            input_zarr, output_zarr, levels_to_copy=levels_to_copy,
            concurrency=self.args["concurrency"])


if __name__ == "__main__":
    mod = ZarrMIPTransferModule()
    mod.run()

    # input_zarr, output_dir = map(pathlib.Path, sys.argv[1:])
    # levels_to_copy = {"1"}

    # # acq_group = zarr.open(input_zarr, mode="r")
    # new_zarr_base = output_dir / input_zarr.name

    # copy_zarr_levels_from_base(
    #     input_zarr, new_zarr_base, levels_to_copy=levels_to_copy)
    # copy_zarr_group(acq_group, new_zarr_base, only_top_files=True)
    # for position_key, position_group in acq_group.items():
    #     copy_zarr_group(position_group, new_zarr_base, only_top_files=True)
    #     for level_key, level_ds in position_group.items():
    #         if level_key in levels_to_copy:
    #             copy_zarr_group(level_ds, new_zarr_base, only_top_files=False)
    #         else:
    #             copy_zarr_group(level_ds, new_zarr_base, only_top_files=True)