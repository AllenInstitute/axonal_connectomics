# Code for generating segmentation & skeleton layers for visualization in neuroglancer.
#
# Segmentation input format:
#   TIFF images stored in the directory ${INPUT}/Segmentation_labeled/
#   Filenames should be zero padded and named NNNN.tiff
#
# Skeleton input format:
#   SWC files stored in the directory ${INPUT}/swc_files_nm
#   Filenames should be zero padded and named NNNN.swc.
#   The integer value of NNNN should match the segmentation label in the segmentation TIFFs.
# 
# Example usage:
#   python segmentation.py --input_dir /Users/eric/nobackup/allen/olga/MN7_RH_3_2_S35_220127_high_res/Pos10_10 --output_dir /Users/eric/nobackup/allen/out/Pos10_10 
#
# TODO: Fix default nm settings & make it configurable
# TODO: Tweak compression & encoding forants; e.g., do we need uint64?
# TODO: Use the sharded format (fewer files)

import numpy as np
import tifffile
import glob
import os
import json

import argschema

from cloudvolume import CloudVolume, Skeleton

def generate_ngl_segmentation(source_path, out_path):
    """Create a neuroglancer precomputed segmentation volume from a tiff stack.
    
       source_path: directory underwhich `swc_files_nm` will be found
                    (Example path: /data/olga/MN7_RH_3_2_S35_220127_high_res/Pos10_10)
       out_path: directory where new the new cloud volume segmentation layer should be generated
                    (Example path: /data/out/MN7_RH_3_2_S35_220127_high_res/Pos10_10)
    """

    # Read tiffstack into memory
    files = glob.glob(f"{source_path}/Segmentation_labeled/*.tif")
    files = sorted(files)
    data = None
    for layer in range(len(files)):
        image = tifffile.imread(files[layer])
        # Allocate array if it has not been yet, using the number of files and dimensions of first file
        if data is None:
            data = np.zeros(shape=(image.shape[1], image.shape[0], len(files)), dtype=image.dtype)
        data[:, :, layer] = image.T   # Swap X & Y to match the SWC files? (not sure why this is needed)
 
    info = CloudVolume.create_new_info(
        num_channels    = 1,
        layer_type      = 'segmentation',
        data_type       = 'uint64', # Channel images might be 'uint8'
        # raw, png, jpeg, compressed_segmentation, fpzip, kempressed, compresso
        encoding        = 'compressed_segmentation', 
        #resolution      = [406, 406, 1997.72], # Voxel scaling, units are in nanometers
        resolution      = [406, 406, 769], # Voxel scaling, units are in nanometers
        voxel_offset    = [0, 0, 0], # x,y,z offset in voxels from the origin
        # Pick a convenient size for your underlying chunk representation
        # Powers of two are recommended, doesn't need to cover image exactly
        chunk_size      = [ 512, 512, 64, ], # units are voxels
        volume_size     = data.shape, # e.g. a cubic millimeter dataset
        skeletons       = 'skeletons'
        )

    vol = CloudVolume(f'file://{out_path}', info=info, compress='', cache=False)
    print("Creating cloud volume: ", vol.info)
    vol.commit_info()
    vol.commit_provenance()
    vol[:,:,:] = data.astype(np.uint64)


def generate_ngl_skeletons(source_path, out_path):
    """Generate skeletons from SWC files.
       This currently assumes the neuroglancer precomputed volume has already been generated by generate_ngl_segmentation.

       source_path: directory underwhich `swc_files_nm` will be found.
       out_path: directory with the previously generated segmentation layer.
       
    """
    # There a few cloud-volume bugs we are working around:
    #   * The info file is not generated

    vol = CloudVolume(f'file://{out_path}', compress='')
    files = glob.glob(f"{source_path}/swc_files_nm/*.swc")
    files = sorted(files)
    skel_dir = os.path.join(out_path, "skeletons")
    if not os.path.exists(skel_dir):
        os.makedirs(skel_dir)
        
    skel_info = {"@type": "neuroglancer_skeletons",}
    with open(os.path.join(skel_dir, "info"), "w") as f:
        json.dump(skel_info, f)
        #f.write('{"@type": "neuroglancer_skeletons"}')
    
    segprops = {"@type": "neuroglancer_segment_properties",
            "inline" : {
                "ids" : [],
                "properties" : [
                    {"id": "tags",
                        "type": "tags",
                        "tags" : ["all"],
                        "values" : []
                    },
                    {"id": "length",
                        "type": "number",
                        "data_type" : "float32",
                        "values" : []
                    }
                ]},
            }
    
    for filename in files:
        # ..../NNNN.swc -> NNNN
        skel_id = int(os.path.split(filename)[-1].split('.')[0])
        with open(filename, mode='r') as f:
            swc = f.read()
        skel = Skeleton.from_swc(swc)
        skel.id = skel_id
        
        skel_out = os.path.join(skel_dir, str(skel_id))
        with open(skel_out, mode="wb") as f:
            f.write(skel.to_precomputed())
 
        segprops["inline"]["ids"].append(str(skel_id))
        segprops["inline"]["properties"][0]["values"].append([0])  # tags
        segprops["inline"]["properties"][1]["values"].append(str(skel.cable_length()))
        
            
    # Write the segment properties
    segment_info_dir = os.path.join(out_path, "segment_properties")
    os.makedirs(segment_info_dir, exist_ok=True)
    with open(os.path.join(segment_info_dir, "info"), "w") as f:
        json.dump(segprops, f)
        
    # Re-write info file with added segment_properties
    with open(f'{out_path}/info', 'r') as f:
        infofile = json.load(f)
    infofile['segment_properties'] = 'segment_properties'
    with open(f'{out_path}/info', 'w') as f:
        json.dump(infofile, f)


class SegmentationToNeuroglancerParameters(argschema.ArgSchema):
    input_dir = argschema.fields.InputDir(required=True)
    output_dir = argschema.fields.InputDir(required=True)


class SegmentationToNeuroglancer(argschema.ArgSchemaParser):
    default_schema = SegmentationToNeuroglancerParameters

    def run(self):
        generate_ngl_segmentation(self.args["input_dir"], self.args["output_dir"])
        generate_ngl_skeletons(self.args["input_dir"], self.args["output_dir"])

if __name__ == "__main__":
    mod = SegmentationToNeuroglancer()
    print(mod.args)
    mod.run()