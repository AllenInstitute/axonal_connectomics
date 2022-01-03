from pathlib import Path
import time
import os

test_input = {
    "position": 1,
    "rootDir": "/ACdata/processed/demoModules/raw/",
    "outputDir": "/ACdata/processed/demoModules/output/",
    'dsName':'ex1'
}
convert_input = {
        "ds_name": "pos1",
        "max_mip": 4,
        "concurrency": 20,
        "input_dir": "/ACdata/processed/demoModules/raw/ex1_Pos1",
        "out_n5": "/ACdata/processed/rmt_testKevin/ispim2/n5/testPos1.n5"
        }
        
start = time.time()

#convert to n5
from acpreprocessing.stitching_modules.convert_to_n5 import tiff_to_n5
mod = tiff_to_n5.TiffDirToN5(input_data=convert_input, args=[])
mod.run()

#test multiscale
from acpreprocessing.stitching_modules.multiscale_viewing  import multiscale
mod = multiscale.Multiscale()
mod.run()

#state = {"layers": []}

#from acpreprocessing.stitching_modules.nglink import create_layer, create_nglink
#for pos in range (3):
#    demo_input = {
#        "position": pos,
#        "outputDir": "/ACdata/processed/demoModules/output/"
#    }
#    mod1 = create_layer.NgLayer(input_json = demo_input)
#    mod1.run(state)
    

#mod2 = create_nglink.Nglink()
#mod2.run(state)

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
