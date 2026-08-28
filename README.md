[![DOI](https://zenodo.org/badge/260070458.svg)](https://doi.org/10.5281/zenodo.22151004)
# axonal_connectomics
Repository for tools developed for axonal connectomics

### Level of Support
We are planning on occasional updating this tool with no fixed schedule. Community involvement is encouraged through both issues and pull requests. Please make pull requests against the develop branch, as we will test changes there before merging into main.

### Methods

**Deskew and zarr conversion**
`acpreprocessing.stitching_modules.convert_to_n5.tiff_to_ngff`
- Sequentially reads image arrays from a tiff stack series for pixel-wise deskew (optional), for computing a downsampling pyramid to a user-defined depth, and for writing out the data volume into a next-generation file format (zarr v3).
```
python zarrv3_to_zarr.py \
--input_file PATH_TO_INPUT_ZARR_FILE \
--output_file PATH_TO_OUTPUT_ZARR_FILE \
--group_names LIST_OF_GROUP_NAMES \
--group_attributes LIST_OF_GROUP_ATTRIBUTE_DICTS \
--block_concurrency SLICE_LEVEL_CONCURRENCY_DEFAULT_1 \
--deskew_options DICT_OF_DESKEW_OPTIONS
```

**Point extraction**
`acpreprocessing.stitching_modules.acstitch.extract_points`
- Detects blob features within user-defined ROIs of a source tile and generates initial point correspondences with a target tile using the estimated tile offset.
```
python extract_points.py \
--p_tile PATH_TO_SOURCE_TILE \
--q_tile PATH_TO_TARGET_TILE \
--output_file PATH_TO_OUTPUT_POINTMATCH_FILE \
--mip_lvl MIP_LEVEL_DEFAULT_0 \
--roi_file PATH_TO_ROI_JSON_FILE \
--method BLOB_DETECTION_METHOD_log_dog_OR_doh \
--blob_kwargs DICT_OF_BLOB_DETECTION_KWARGS \
--n_points NUMBER_OF_POINTS_PER_ROI_DEFAULT_1
```

**Tile stitching**
`acpreprocessing.stitching_modules.acstitch.stitch`
- Generate point correspondences between tiles from template matching or SIFT features at user-defined resolution level (mip).
```
python stitch_tiles.py \
--input_file PATH_TO_INPUT_POINTMATCH_FILE \
--output_file PATH_TO_OUTPUT_POINTMATCH_FILE \
--stitch_method STITCH_METHOD_ccorr_OR_sift \
--miplvl MIP_LEVEL_DEFAULT_0 \
--sift_kwargs DICT_OF_SIFT_KWARGS \
--stitch_kwargs DICT_OF_STITCH_KWARGS
```
