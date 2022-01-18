from pathlib import Path
import time
import os
from acpreprocessing.utils import io
from acpreprocessing.stitching_modules.metadata import parse_metadata
from acpreprocessing.stitching_modules.convert_to_n5 import tiff_to_n5
from acpreprocessing.stitching_modules.multiscale_viewing  import multiscale
from acpreprocessing.stitching_modules.nglink import create_layer, create_nglink
from acpreprocessing.stitching_modules.stitch import create_json, stitch

start = time.time()

outputdir = "/ACdata/processed/MN7_RH_3_b5_S16_1_high_res_region"
rootdir = "/ispim2_data/MN7_RH_3_b5_S16_1_high_res_region/"
ds_name = "high_res_region"

md_input = {
        "rootDir":rootdir,
        "fname":"acqinfo_metadata.json"
        }
metadata = parse_metadata.ParseMetadata(input_data=md_input)
n_pos = metadata.get_number_of_positions()

for pos in range(n_pos):
    convert_input = {
        "ds_name": f"pos{pos}",
        "max_mip": 4,
        "concurrency": 20,
        "input_dir": f"{rootdir}{ds_name}_Pos{pos}",
        "out_n5": f"{outputdir}Pos{pos}.n5"
        }
    multiscale_input = {
        "position": pos,
        "outputDir": f"{outputdir}Pos{pos}.n5",
        'max_mip':4,
        "rootDir":f"{rootdir}",
        "fname":"acqinfo_metadata.json"
        }
        
    #Convert to n5
    #print(f"Converting from rootdir: {convert_input["input_dir"]}")
    mod = tiff_to_n5.TiffDirToN5(input_data=convert_input, args=[])
    mod.run()

    #Add multiscale attributes
    mod1 = multiscale.Multiscale(input_data = multiscale_input)
    mod1.run()

#Create overview nglink
state = {"layers": []}

for pos in range (n_pos):
    layer_input = {
        "position": pos,
        "outputDir": outputdir,
        "rootDir": rootdir
        }
    mod2 = create_layer.NgLayer(input_data = layer_input)
    mod2.run(state)

nglink_input={
        "outputDir": outputdir,
        "fname": "nglink.txt"
        }
create_nglink.Nglink(input_data=nglink_input).run(state)


create_json_input = {
        'rootDir':rootdir,
        'outputDir':outputdir
        }
mod = create_json.CreateJson(input_data = create_json_input)
mod.run()

stitchjson=os.path.join(outputdir,"stitch.json")
stitch_input={
        "stitchjson":stitchjson
        }
mod = stitch.Stitch(input_data=stitch_input)
mod.run()

statejson = io.read_json(os.path.join(outputdir,"state.json"))
#read stitch json in
stitchoutjson = io.read_json(os.path.join(outputdir,"stitch-final.json"))
    
#update state json with stich coord
for pos in range(0,n_pos-1):
    try:
        statejson['layers'][pos]['source']['transform']['matrix'][0][3]=stitchoutjson[pos]['position'][0]*4
        statejson['layers'][pos]['source']['transform']['matrix'][1][3]=stitchoutjson[pos]['position'][1]*4
        statejson['layers'][pos]['source']['transform']['matrix'][2][3]=stitchoutjson[pos]['position'][2]*4
    except:
        print("Something went wrong with the stitching output!")

#create stitched nglink
nglink_input={
        "outputDir": outputdir,
        "fname": "stitched-nglink.txt"
        }
create_nglink.Nglink(input_data=nglink_input).run(statejson)

#save new state
io.save_metadata(os.path.join(outputdir,"stitched-state.json"),statejson)

print('done testing')
print(time.time()-start)
