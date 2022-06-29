from pathlib import Path
import time
import os
from urllib import response
import numpy as np
import json
from acpreprocessing.utils import io
from acpreprocessing.utils.metadata import parse_metadata
from acpreprocessing.stitching_modules.convert_to_n5 import acquisition_dir_to_n5_dir
from acpreprocessing.stitching_modules.multiscale_viewing  import multiscale
from acpreprocessing.utils.nglink import create_layer, create_nglink, update_state
from acpreprocessing.stitching_modules.stitch import create_json, stitch
import argschema
from argschema.fields import NumpyArray, Boolean, Float, Int, Str
from ddbclient import acquisition


def find_entry(response_json):
    for entry in response_json:
        uri = entry["data_location"]["ACTiff_dir_ispim"]["uri"]
        if uri.find("high_res") != -1:
            return [uri,entry["scope"]]
        return []


start = time.time()

# ping db to find new samples to process
# acq_client = acquisition.AcquisitionClient("http://bigkahuna.corp.alleninstitute.org/transfer_temp/api/api", subpath="")
# response_json = acq_client.query(
#     {
#         "filter": {"qc_state": "pass"},
#         "projection": {"_id": False}
#     })
# print(response_json)

# data_loc = "/ACdata/workflow_data/workflow_tiff_dirs_by_scope/iSPIM2/MN7_RH_3_9_S33_220421_high_res/"

# # for each sample:
# with open(data_loc+'dbpost.json', 'r') as f:
#     db_md = json.load(f)
#     spec_id = db_md["specimen_id"]
#     sess_id = db_md["session_id"]
#     sec_id = db_md["section_id"]

# acq_client = acquisition.AcquisitionClient("http://bigkahuna.corp.alleninstitute.org/transfer_temp/api/api", subpath="")
# response_json = acq_client.query(
#     {
#         "filter": {"specimen_id": spec_id,"session_id": sess_id, 'section_num': sec_id},
#         "projection": {"_id": False}
#     })
# if not response_json:
#     print("Error: Could not find acquisition")
# else:
#     info = find_entry(response_json)
#     # print(info)
#     if not info:
#         "Error: Unable to find high_res data location"
#     else:
#         dir_ACdata = info[0]
#         rootDir = "/ACdata"+dir_ACdata.split(':')[1]
#         # print(rootDir)
#         scope = info[1]
#         dirname = rootDir.split("/")[-1]
#         print(dirname)

# run_input = {
#     "outputDir": "/ACdata/processed/"+scope+"/"+dirname+"/",
#     "rootDir": rootDir + "/",
#     "ds_name": dirname,
#     "mip_level": 3,
#     "md_filename": rootDir+"/acqinfo_metadata.json",
#     "consolidate_pos": True,
#     "reverse_stitch": False,
#     "deskew": False
# }

run_input = {
    "outputDir": "/ACdata/processed/iSPIM2/MN7_RH_3_9_S19_220606_high_res/",
    "rootDir": "/ispim2_data/workflow_data/iSPIM2/MN7_RH_3_9_S19_220606_high_res/",
    "ds_name": "MN7_RH_3_9_S19_220606_high_res",
    "mip_level": 3,
    "md_filename": "/ispim2_data/workflow_data/iSPIM2/MN7_RH_3_9_S19_220606_high_res/acqinfo_metadata.json",
    "consolidate_pos": True,
    "reverse_stitch": False,
    "deskew": False,
    "stitch_channel": 0,
    "stitch_json": "stitch-test.json",
    "stitch_final": "stitch-final.json",
    "state_json": "state-test.json",
    "nglink_name": "nglink-test.txt",
    "stitched_nglink": "stitched-nglink-test.txt",
    "stitched_state": "stitched-state.json"
}
dirname="MN7_RH_3_9_S19_220606_high_res"
print(run_input)

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
    stitch_channel = Int(required=False,default=0, description="Which channel to compute stitching with")
    stitch_json = Str(required=False,default="stitch.json", description="Name of stitching parameters json file")
    stitch_final = Str(required=False,default="stitch-final.json", description="Name of final stitching result json file")
    state_json = Str(required=False,default="state.json", description="Name of overview state json file")
    nglink_name = Str(required=False,default="nglink.txt", description="Name of nglink txt file")
    stitched_nglink = Str(required=False,default="stitched-nglink.txt", description="Name of stitched nglink txt file")
    stitched_state = Str(required=False,default="stitched-state.json", description="Name of stitched state json file")


class RunModules(argschema.ArgSchemaParser):
    default_schema = RunModulesSchema

    def run(self):

        md_input = {
                "rootDir": run_input['rootDir'],
                "fname": run_input['md_filename']
                }
        md = parse_metadata.ParseMetadata(input_data=md_input).get_md()
        # dsname = md["stripdirs"][0].split("_Pos")[0]
        # run_input["ds_name"]=dsname
        if (md["positions"][0]["y_start_um"])>(md["positions"][1]["y_start_um"]):
            run_input["reverse_stitch"] = True
        deskew = 0
        if run_input['deskew']:
            deskew = np.cos(parse_metadata.ParseMetadata(input_data=md_input).get_angle())
        n_channels = md["channels"]
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

        # Create overview nglink, initialize state for each channel
        state = {"showDefaultAnnotations": False, "layers": []}
        for channel in range(n_channels):
            for pos in range(n_pos):
                if run_input["consolidate_pos"]:
                    layer_input = {
                            "position": 0,
                            "outputDir": run_input['outputDir']+dirname+".n5/channel"+str(channel),
                            "rootDir": run_input['rootDir'],
                            "reverse": run_input["reverse_stitch"],
                            "deskew": deskew,
                            "channel": channel
                            }
                    create_layer.NgLayer(input_data=layer_input).run_consolidate(state)
                    break
                else:
                    layer_input = {
                        "position": pos,
                        "outputDir": run_input['outputDir']+dirname+".n5/channel"+str(channel),
                        "rootDir": run_input['rootDir'],
                        "reverse": run_input["reverse_stitch"],
                        "deskew": deskew,
                        "channel": channel
                        }
                    create_layer.NgLayer(input_data=layer_input).run(state)

        # Create nglink from created state and estimated positions (overview link)
        nglink_input = {
                "outputDir": run_input['outputDir'],
                "fname": run_input["nglink_name"],
                "state_json": run_input["state_json"]
                }
        if not os.path.exists(os.path.join(nglink_input['outputDir'], nglink_input['fname'])):
            create_nglink.Nglink(input_data=nglink_input).run(state)
        else:
            print("nglink txt file already exists!")

        # Create Stitch.json which is input for stitching code (using channel 0)
        create_json_input = {
                'rootDir': run_input['rootDir']+"/",
                'outputDir': run_input['outputDir']+"/",
                "mip_level": run_input['mip_level'],
                "reverse": run_input['reverse_stitch'],
                "dirname": dirname,
                "stitch_channel": run_input['stitch_channel'],
                "stitch_json": run_input['stitch_json']
                }
        create_json.CreateJson(input_data=create_json_input).run()

        # Run Stitching with stitch.json
        stitchjsonpath = os.path.join(create_json_input['outputDir'], run_input['stitch_json'])
        stitch_input = {
                "stitchjson": stitchjsonpath
                }
        # Perform stitching if not done yet
        if not os.path.exists(os.path.join(run_input['outputDir'], run_input["stitch_final"])):
            stitch.Stitch(input_data=stitch_input).run()
        else:
            print("Skipped stitching - already computed")

        # update state json with stitched coord
        update_state_input = {
            'rootDir': run_input['rootDir'],
            "outputDir": run_input['outputDir'],
            'mip_level': run_input['mip_level'],
            "fname": "stitched-nglink.txt",
            "consolidate_pos": run_input['consolidate_pos'],
            "n_channels": n_channels,
            "state_json": run_input["state_json"],
            "stitch_final": run_input["stitch_final"],
            "stitched_nglink": run_input["stitched_nglink"],
            "stitched_state": run_input["stitched_state"]
        }
        update_state.UpdateState(input_data=update_state_input).run()

        print('Done processing')
        print(str((time.time()-start)/60) + " minutes elapsed")


if __name__ == '__main__':
    RunModules(run_input).run()
