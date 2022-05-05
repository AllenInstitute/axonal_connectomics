from pathlib import Path
import time
import os
import numpy as np
from acpreprocessing.utils import io
from acpreprocessing.stitching_modules.metadata import parse_metadata
from acpreprocessing.stitching_modules.convert_to_n5 import acquisition_dir_to_n5_dir
from acpreprocessing.stitching_modules.multiscale_viewing  import multiscale
from acpreprocessing.stitching_modules.nglink import create_layer, create_nglink, update_state
from acpreprocessing.stitching_modules.stitch import create_json, stitch
import argschema
from argschema.fields import NumpyArray, Boolean, Float, Int, Str
from ddbclient import acquisition

start = time.time()

#TODO: make this more robust
acq_client = acquisition.AcquisitionClient("http://bigkahuna.corp.alleninstitute.org/transfer_temp/api/api", subpath="")
response_json = acq_client.query(
    {
        "filter": {"specimen_id": "MN6_2_a","session_id":"220427", 'section_num': '79'},
        "projection": {"_id": False}
    })
#TODO: how to get index of correct data? overview, high_res, etc
dir_ACdata=response_json[3]["data_location"]["ACTiff_dir_ACdata"]["uri"]
rootDir = "/ACdata"+dir_ACdata.split(':')[1]
scope = response_json[0]['scope']
dirname = rootDir.split("/")[5]

run_input = {
    "outputDir": "/ACdata/processed/"+scope+"/"+dirname+"/",
    "rootDir": rootDir + "/",
    "ds_name": "",
    "mip_level": 3,
    "md_filename": rootDir+"/acqinfo_metadata.json",
    "consolidate_pos": True,
    "reverse_stitch": False,
    "deskew": False
}


class RunModulesSchema(argschema.ArgSchema):
    outputDir = Str(required=True, description='output directory')
    rootDir = Str(required=True, description='raw tiff root directory')
    ds_name = Str(required=True)
    mip_level = Int(required=False, default=3,
                    description='downsample level to perform stitching with')
    md_filename = Str(required=False, default="acqinfo_metadata.json",
                      description='metadata file name')
    consolidate_pos = Boolean(required=False, default=True,
                              description="Whether to consolidate all position"
                                          "strips in one neuroglancer layer or"
                                          "separate them into independent layers")
    reverse_stitch = Boolean(required=False,default=False, description="Whether to reverse direction of stitching or not")
    deskew = Boolean(required=False,default=False, description="Whether to deskew or not")


class RunModules(argschema.ArgSchemaParser):
    default_schema = RunModulesSchema

    def run(self):

        md_input = {
                "rootDir": run_input['rootDir'],
                "fname": run_input['md_filename']
                }
        md = parse_metadata.ParseMetadata(input_data=md_input).get_md()
        dsname = md["stripdirs"][0].split("_Pos")[0]
        run_input["ds_name"]=dsname
        if (md["positions"][0]["y_start_um"])>(md["positions"][1]["y_start_um"]):
            run_input["reverse_stitch"] = True
        print(run_input)
        
        deskew = 0
        if run_input['deskew']:
            deskew = np.cos(parse_metadata.ParseMetadata(input_data=md_input).get_angle())
        
        n_pos = parse_metadata.ParseMetadata(input_data=md_input).get_number_of_positions()
        if n_pos == 0:
            print("No positions to process")
            return -1

        convert_input = {
            "input_dir": f"{run_input['rootDir']}",
            "output_dir": f"{run_input['outputDir']}",
            "max_mip": 5,
            "position_concurrency": 5         
            }
        if not os.path.isdir(convert_input['output_dir']):
            acquisition_dir_to_n5_dir.AcquisitionDirToN5Dir(input_data=convert_input).run()
        else:
            print(f"Skipping conversion, {dirname} directory already exists")
    
        for pos in range(n_pos):
        # Create overview nglink, initialize state
            state = {"showDefaultAnnotations": False, "layers": []}

            if run_input["consolidate_pos"]:
                layer_input = {
                        "position": 0,
                        "outputDir": run_input['outputDir']+dirname+".n5/",
                        "rootDir": run_input['rootDir'],
                        "reverse": run_input["reverse_stitch"],
                        "deskew": deskew
                        }
                create_layer.NgLayer(input_data=layer_input).run_consolidate(state)
            else:
                # Loop throuh each position to add a layer to state
                for pos in range(n_pos):
                    layer_input = {
                        "position": pos,
                        "outputDir": run_input['outputDir']+dirname+".n5/",
                        "rootDir": run_input['rootDir'],
                        "reverse": run_input["reverse_stitch"],
                        "deskew": deskew
                        }
                    create_layer.NgLayer(input_data=layer_input).run(state)

        # Create nglink from created state and estimated positions (overview link)
        nglink_input = {
                "outputDir": run_input['outputDir'],
                "fname": "nglink.txt"
                }
        if not os.path.exists(os.path.join(run_input['outputDir'],"nglink.txt")):
            create_nglink.Nglink(input_data=nglink_input).run(state)
        else:
            print("nglink.txt already exists!")

        # Create Stitch.json which is input for stitching code
        create_json_input = {
                'rootDir': run_input['rootDir'],
                'outputDir': run_input['outputDir'],
                "mip_level": run_input['mip_level'],
                "reverse": run_input['reverse_stitch'],
                "dirname": dirname
                }
        create_json.CreateJson(input_data=create_json_input).run()

        # Run Stitching with stitch.json
        stitchjsonpath = os.path.join(run_input['outputDir'], "stitch.json")
        stitch_input = {
                "stitchjson": stitchjsonpath
                }
        # Perform stitching if not done yet
        if not os.path.exists(os.path.join(run_input['outputDir'],"stitch-final.json")):
            stitch.Stitch(input_data=stitch_input).run()
        else:
            print("Skipped stitching - already computed")

        # update state json with stitched coord
        update_state_input = {
            'rootDir': run_input['rootDir'],
            "outputDir": run_input['outputDir'],
            'mip_level': run_input['mip_level'],
            "fname": "stitched-nglink.txt",
            "consolidate_pos": run_input['consolidate_pos']
        }
        update_state.UpdateState(input_data=update_state_input).run()

        print('Done processing')
        print(str((time.time()-start)/60) + " minutes elapsed")


if __name__ == '__main__':
    RunModules(run_input).run()
