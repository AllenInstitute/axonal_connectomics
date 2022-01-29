from pathlib import Path
import time
import os
from acpreprocessing.utils import io
from acpreprocessing.stitching_modules.metadata import parse_metadata
from acpreprocessing.stitching_modules.convert_to_n5 import tiff_to_n5
from acpreprocessing.stitching_modules.multiscale_viewing  import multiscale
from acpreprocessing.stitching_modules.nglink import create_layer, create_nglink, update_state
from acpreprocessing.stitching_modules.stitch import create_json, stitch
import argschema
from argschema.fields import NumpyArray, Boolean, Float, Int, Str

start = time.time()

run_input = {
    "outputDir": "/ACdata/processed/MN7_RH_3_2_s27_high_res/",
    "rootDir": "/ispim2_data/MN7_RH_3_2_s27_high_res/",
    "ds_name": "ex1_1",
    "mip_level": 3,
    "md_filename": "acqinfo_metadata.json"
}


class RunModulesSchema(argschema.ArgSchema):
    outputDir = Str(required=True, description='output directory')
    rootDir = Str(required=True, description='raw tiff root directory')
    ds_name = Str(required=True)
    mip_level = Int(required=False, default=3,
                    description='downsample level to perform stitching with')
    md_filename = Str(required=False, default="acqinfo_metadata.json",
                      description='metadata file name')


class RunModules(argschema.ArgSchemaParser):
    default_schema = RunModulesSchema

    def run(self):

        md_input = {
                "rootDir": run_input['rootDir'],
                "fname": run_input['md_filename']
                }
        n_pos = parse_metadata.ParseMetadata(input_data=md_input).get_number_of_positions()

        for pos in range(n_pos):
            convert_input = {
                "ds_name": f"pos{pos}",
                "max_mip": 4,
                "concurrency": 20,
                "input_dir": f"{run_input['rootDir']}/{run_input['ds_name']}_Pos{pos}",
                "out_n5": f"{run_input['outputDir']}/Pos{pos}.n5"
                }
            multiscale_input = {
                "position": pos,
                "outputDir": f"{run_input['outputDir']}/Pos{pos}.n5",
                'max_mip': 4,
                "rootDir": f"{run_input['rootDir']}",
                "fname": "acqinfo_metadata.json"
                }

            # Convert to n5
            tiff_to_n5.TiffDirToN5(input_data=convert_input, args=[]).run()
            # Add multiscale attributes
            multiscale.Multiscale(input_data=multiscale_input).run()

        # Create overview nglink, initialize state
        state = {"showDefaultAnnotations": False, "layers": []}

        # Loop throuh each position to add a layer to state
        for pos in range(n_pos):
            layer_input = {
                "position": pos,
                "outputDir": run_input['outputDir'],
                "rootDir": run_input['rootDir']
                }
            create_layer.NgLayer(input_data=layer_input).run(state)

        # Create nglink from created state and estimated positions (overview link)
        nglink_input = {
                "outputDir": run_input['outputDir'],
                "fname": "nglink.txt"
                }
        create_nglink.Nglink(input_data=nglink_input).run(state)

        # Create Stitch.json which is input for stitching code
        create_json_input = {
                'rootDir': run_input['rootDir'],
                'outputDir': run_input['outputDir'],
                "mip_level": run_input['mip_level']
                }
        create_json.CreateJson(input_data=create_json_input).run()

        # Run Stitching with stitch.json
        stitchjsonpath = os.path.join(run_input['outputDir'], "stitch.json")
        stitch_input = {
                "stitchjson": stitchjsonpath
                }
        stitch.Stitch(input_data=stitch_input).run()

        # update state json with stitched coord
        update_state_input = {
            'rootDir': run_input['rootDir'],
            "outputDir": run_input['outputDir'],
            'mip_level': run_input['mip_level'],
            "fname": "stitched-nglink.txt"
        }
        update_state.UpdateState(input_data=update_state_input).run()

        print('Done processing')
        print(str((time.time()-start)/60) + " minutes elapsed")


if __name__ == '__main__':
    RunModules(run_input).run()
