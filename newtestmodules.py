from pathlib import Path
import time
import os

multiscale_input = {
    "position": 1,
    "outputDir": "/ACdata/processed/MN7_RH_3_b5_S16_1_high_res_region/Pos1.n5",
    'max_mip':4
}
convert_input = {
        "ds_name": "pos1",
        "max_mip": 4,
        "concurrency": 20,
        "input_dir": "/ispim2_data/MN7_RH_3_b5_S16_1_high_res_region/high_res_region_Pos1",
        "out_n5": "/ACdata/processed/MN7_RH_3_b5_S16_1_high_res_region/Pos1.n5"
        }
        
start = time.time()

#convert to n5
#from acpreprocessing.stitching_modules.convert_to_n5 import tiff_to_n5
#mod = tiff_to_n5.TiffDirToN5(input_data=convert_input, args=[])
#mod.run()

#test multiscale
#from acpreprocessing.stitching_modules.multiscale_viewing  import multiscale
#mod = multiscale.Multiscale(input_data = multiscale_input)
#mod.run()

state = {"layers": []}

from acpreprocessing.stitching_modules.nglink import create_layer, create_nglink
for pos in range (9):
    ng_input = {
        "position": pos,
        "outputDir": "/ACdata/processed/MN7_RH_3_b5_S16_1_high_res_region",
        "rootDir": "/ispim2_data/MN7_RH_3_b5_S16_1_high_res_region"
        }
    mod1 = create_layer.NgLayer(input_data = ng_input)
    mod1.run(state)
    

nglink_input={
        "outputDir": "/ACdata/processed/MN7_RH_3_b5_S16_1_high_res_region",
        "fname": "nglink.txt"
        }
mod2 = create_nglink.Nglink(nglink_input)
mod2.run(state)

#from acpreprocessing.stitching_modules.stitch import create_json
#mod = create_json.CreateJson()
#outputDir = "/ACdata/processed/demoModules/output/"
#mod.run(outputDir)

#from acpreprocessing.stitching_modules.stitch import stitch
#mod = stitch.Stitch()
#outputDir = "/ACdata/processed/demoModules/output/"
#stitchjson = outputDir + "stitch.json"
#mod.run(stitchjson)


print('done testing')
print(time.time()-start)
