from acpreprocessing.stitching_modules.convert_to_2D_tiff import convert_to_2D_tiff
from acpreprocessing.stitching_modules.convert_to_n5 import convert_to_n5
from acpreprocessing.stitching_modules.downsample_n5 import downsample_n5
from acpreprocessing.stitching_modules.metadata  import parse_metadata
from acpreprocessing.stitching_modules.multiscale_viewing  import multiscale
from acpreprocessing.stitching_modules.nglink import create_layer, create_nglink
from acpreprocessing.stitching_modules.stitch import create_json, stitch
from acpreprocessing.utils import io

import time
start = time.time()
mod0 = parse_metadata.ParseMetadata()
n_pos = mod0.get_number_of_positions()

state = {"layers": []}
for pos in range (n_pos):
    run_input = {
    "position": pos,
    "rootDir": "/ispim2_data/MN_6_2_S14_4_ex1/",
    "outputDir": "/ACdata/processed/test_MN_6_2_S14_4_ex1/",
    'dsName':'ex1'
    }

    #convert to 2d tiff
    mod1 = convert_to_2D_tiff.Convert2DTiff(input_json = run_input)
    mod1.run()

    #convert to n5
    mod2 = convert_to_n5.Convert2N5(input_json = run_input)
    mod2.run()

    #downsample n5
    mod3 = downsample_n5.DownsampleN5(input_json = run_input)
    mod3.run()

    #test multiscale
    mod4 = multiscale.Multiscale(input_json = run_input)
    mod4.run()

    mod5 = create_layer.NgLayer(input_json = run_input)
    mod5.run(state)

mod6 = create_nglink.Nglink(input_json = run_input)
mod6.run(state, fname='nglink.txt')

#save state json
io.save_metadata(run_input['outputDir']+"state.json",state)

#stitch
mod7 = create_json.CreateJson(input_json = run_input)
mod7.run()

mod = stitch.Stitch()
mod.run(run_input['outputDir'] + "stitch.json")

#add stitch coord to nglink
#read state json in
statejson = io.read_json(run_input['outputDir'] + "state.json")
#read stitch json in
stitchoutjson = io.read_json(run_input['outputDir'] + "stitch-final.json")
#update state json with stich coord
for pos in range(n_pos):
    statejson['layers'][pos]['source']['transform']['matrix'][0][3]=stitchoutjson[pos]['position'][0]*4
    statejson['layers'][pos]['source']['transform']['matrix'][1][3]=stitchoutjson[pos]['position'][1]*4
    statejson['layers'][pos]['source']['transform']['matrix'][2][3]=stitchoutjson[pos]['position'][2]*4
#create stitched nglink
mod6.run(statejson, fname='stitched-nglink.txt')

#save new state
io.save_metadata(run_input['outputDir']+"stitched-state.json",statejson)

print('done running modules')
print(time.time()-start)
