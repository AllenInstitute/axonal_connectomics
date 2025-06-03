"""
This script provides utilities to extract volumetric cutouts from Zarr datasets using Neuroglancer links.
It supports:
- Parsing Neuroglancer state JSON from a link.
- Resolving Zarr dataset URLs and drivers from Neuroglancer layers.
- Computing voxel-space translations based on dataset metadata and transformations.
- Downloading and saving specified subvolumes as TIFF files, with associated metadata.

Typical usage:
    python cutout.py --ng_link <NG_LINK> --layer_name <LAYER> --center Z Y X --size Z Y X [--out_path <PATH>] [--mip <MIP>] [--offset Z Y X]

Dependencies: tensorstore, numpy, requests, tifffile

TODO: check if exisiting utilities can be used to replace some of the functionality here.

Currently there is a standalone repo that includes this tool: https://github.com/AllenInstitute/ac_data_tools

Author: @wanqingy
Date: 2025-06-03
"""


import tensorstore as ts
import numpy as np
import os
import json
from urllib.parse import urljoin
from urllib.request import urlopen
import requests
import tifffile as tiff
import datetime

def get_json_state_from_ng_link(ng_link):
    """
    Fetch the JSON state from a Neuroglancer link.
    If the link contains '#!', extract the JSON state URL after it.
    """
    # If the link contains '#!', extract the part after it
    if '#!' in ng_link:
        ng_link = ng_link.split('#!')[1]
    response = requests.get(ng_link)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to fetch JSON from {ng_link}")
    
def get_layer_translation_voxel(zarr_url, ng_state, layer_name):
    """
    Compute translation_voxel for a given layer name using zarr_url and ng_state.
    Loads zattrs from the given zarr_url.
    """
    # Load zattrs from zarr_url
    if zarr_url.startswith("http"):
        json_url = urljoin(zarr_url.rstrip("/") + "/", "zarr.json")
        try:
            with urlopen(json_url) as f:
                zattrs = json.load(f)
        except Exception:
            zattrs_url = urljoin(zarr_url.rstrip("/") + "/", ".zattrs")
            with urlopen(zattrs_url) as f:
                zattrs = json.load(f)
    else:
        json_path = os.path.join(zarr_url, "zarr.json")
        try:
            with open(json_path, "r") as f:
                zattrs = json.load(f)
        except Exception:
            zattrs_path = os.path.join(zarr_url, ".zattrs")
            with open(zattrs_path, "r") as f:
                zattrs = json.load(f)

    # Get scale for highest resolution (first dataset)
    if "attributes" in zattrs and "multiscales" in zattrs["attributes"]:
        multiscale = zattrs['attributes']["multiscales"][0]
    elif "multiscales" in zattrs:
        multiscale = zattrs["multiscales"][0]
    else:
        multiscale = {}

    # Get spatial scale (z, y, x) for highest resolution
    if "datasets" in multiscale and multiscale["datasets"]:
        first_dataset = multiscale["datasets"][0]
        scale = None
        for t in first_dataset.get("coordinateTransformations", []):
            if t.get("type") == "scale":
                scale = t["scale"]
                break
        if scale is not None:
            spatial_scale = np.array(scale[2:5])
        else:
            spatial_scale = np.array([1.0, 1.0, 1.0])
    else:
        spatial_scale = np.array([1.0, 1.0, 1.0])

    # Get translation_mip0 from multiscale
    translation_mip0 = [0.0, 0.0, 0.0]
    for t in multiscale.get("coordinateTransformations", []):
        if t.get("type") == "translation":
            translation_mip0 = t.get("translation", [0.0, 0.0, 0.0])[-3:]
            break

    translation_voxel = np.array(translation_mip0) / spatial_scale

    # --- Now check for transform matrix in the layer by name ---
    layer = None
    for lyr in ng_state.get('layers', []):
        if lyr.get('name') == layer_name:
            layer = lyr
            break
    if layer is None:
        raise ValueError(f"Layer with name '{layer_name}' not found in ng_state.")

    transform_matrix = None
    src = layer.get('source')
    if isinstance(src, dict):
        transform = src.get('transform')
        if transform and 'matrix' in transform:
            transform_matrix = np.array(transform['matrix'])
    elif 'transform' in layer:
        transform = layer['transform']
        if 'matrix' in transform:
            transform_matrix = np.array(transform['matrix'])

    # If present, add the translation from the matrix (last element of each row, z/y/x)
    if transform_matrix is not None:
        # For 5x6 matrix, translation is last element of each row (z, y, x order)
        translation_from_matrix = np.array([
            transform_matrix[2, -1],  # z
            transform_matrix[3, -1],  # y
            transform_matrix[4, -1],  # x
        ])
        translation_voxel += translation_from_matrix

    return translation_voxel.astype(int)

def get_zarr_url_and_driver_from_ng_state(ng_state, layer_name):
    """
    Given a Neuroglancer state (already loaded as dict) and a layer name,
    return a tuple (zarr_url, driver) for that layer.
    - zarr_url: always starts from 'http', with any prefix (e.g. 'zarr://') and any suffix (e.g. '|zarr3:') removed.
    - driver: parsed from the suffix after '|', e.g. 'zarr3', or defaults to 'zarr3' if not found.
    """
    for layer in ng_state.get('layers', []):
        if layer.get('name') == layer_name:
            src = layer.get('source')
            url = None
            # If source is a string
            if isinstance(src, str):
                url = src
            # If source is a dict with 'url'
            elif isinstance(src, dict) and 'url' in src:
                url = src['url']
            # If source is a list, return the first string or dict with 'url'
            elif isinstance(src, list):
                for s in src:
                    if isinstance(s, str):
                        url = s
                        break
                    elif isinstance(s, dict) and 'url' in s:
                        url = s['url']
                        break
            if url is None:
                raise ValueError(f"Could not find a valid zarr url in source for layer '{layer_name}'")
            # Find the start of the http/https URL
            idx = url.find('http')
            if idx != -1:
                url_clean = url[idx:]
            else:
                url_clean = url
            # Remove any suffix after the first whitespace or '|' (if present)
            driver = 'zarr3'
            if '|' in url_clean:
                parts = url_clean.split('|', 1)
                url_clean = parts[0]
                driver_part = parts[1]
                if ':' in driver_part:
                    driver_candidate = driver_part.split(':')[0]
                    if driver_candidate:
                        driver = driver_candidate
            elif ' ' in url_clean:
                url_clean = url_clean.split(' ')[0]
            return url_clean, driver
    raise ValueError(f"Layer with name '{layer_name}' not found in NG state.")

def download_cutout_from_zarr(
    ng_link,
    layer_name,
    center,
    size,
    out_path=None,
    post_fix=None,
    mip="0",
    offset=None
):
    print("Opening TensorStore...")

    ng_state = get_json_state_from_ng_link(ng_link)
    zarr_url, driver = get_zarr_url_and_driver_from_ng_state(ng_state, layer_name)
    spec = {
        "driver": driver,
        "kvstore": zarr_url,
        "path": mip,
    }

    store = ts.open(spec).result()
    domain = store.domain
    zyx_min = domain.inclusive_min[2:]
    zyx_max = tuple(x - 1 for x in domain.exclusive_max[2:])
    zyx_shape = domain.shape[2:]

    if offset is None:
        offset = get_layer_translation_voxel(zarr_url, ng_state, layer_name)
    else:
        offset = np.array(offset)

    center_local = (np.array([int(c - o) for c, o in zip(center, offset)]) // (2 ** int(mip))).astype(int)
    size = [int(s // (2 ** int(mip))) for s in size]

    spatial_start = [int(max(mins, c - s // 2)) for c, s, mins in zip(center_local, size, zyx_min)]
    spatial_stop = [int(min(c + s // 2, maxs)) for c, s, maxs in zip(center_local, size, zyx_max)]

    final_shape = [stop - start for start, stop in zip(spatial_start, spatial_stop)]

    print(f"Global center={center}, Offset={offset}, Local center={center_local}")
    print(f"Clipped start={spatial_start}, stop={spatial_stop}, shape={final_shape}")
    print(f"Dataset shape: zyx = {zyx_shape}")

    cutout = store[0, 0,
                   spatial_start[0]:spatial_stop[0],
                   spatial_start[1]:spatial_stop[1],
                   spatial_start[2]:spatial_stop[2]].read().result()

    # Generate file name if not provided
    parts = zarr_url.strip('/').split('/')
    sample = parts[-3] if len(parts) >= 3 else "sample"
    pos = parts[-1] if len(parts) >= 1 else "Pos"
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if post_fix is None:
        base_name = f"{sample}_{pos}_{now}"
    else:
        base_name = f"{sample}_{pos}_{post_fix}"

    if out_path is None:
        out_path = base_name + ".tiff"
    if not out_path.endswith(".tiff"):
        out_path = out_path + '/' + base_name + ".tiff"

    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    print(f"Saving cutout to {out_path} with shape {cutout.shape}")
    tiff.imwrite(out_path, cutout)

    # Save metadata
    metadata = {
        "NG_link": ng_link,
        "src_loc": zarr_url,
        "image_tile": layer_name,
        "center": center,
        "size": size,
        "mip": mip
    }
    meta_path = os.path.splitext(out_path)[0] + "_metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata, f)

    return cutout

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download a cutout from a Zarr dataset using a Neuroglancer link.")
    parser.add_argument("--ng_link", type=str, required=True, help="Neuroglancer link")
    parser.add_argument("--layer_name", type=str, required=True, help="Layer name in the Neuroglancer state")
    parser.add_argument("--center", type=int, nargs=3, required=True, metavar=("Z", "Y", "X"), help="Center coordinate (z y x)")
    parser.add_argument("--size", type=int, nargs=3, required=True, metavar=("Z", "Y", "X"), help="Size of the cutout (z y x)")
    parser.add_argument("--out_path", type=str, default=None, help="Output TIFF file path")
    parser.add_argument("--mip", type=str, default="0", help="MIP level (default: 0)")
    parser.add_argument("--offset", type=int, nargs=3, metavar=("Z", "Y", "X"), help="Optional offset (z y x)")

    args = parser.parse_args()

    download_cutout_from_zarr(
        ng_link=args.ng_link,
        layer_name=args.layer_name,
        center=args.center,
        size=args.size,
        out_path=args.out_path,
        mip=args.mip,
        offset=args.offset
    )