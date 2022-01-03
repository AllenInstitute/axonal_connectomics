import time
#dem0_input = {
#    "position": 1,
#    "rootDir": "/ACdata/processed/demoModules/raw/",
#    "outputDir": "/ACdata/processed/demoModules/output/",
#    'dsName':'ex1'
#}
start = time.time()
#convert to 2d tiff
#from acpreprocessing.stitching_modules.convert_to_2D_tiff import convert_to_2D_tiff
#mod = convert_to_2D_tiff.Convert2DTiff()
#mod.run()

#convert to n5
#from acpreprocessing.stitching_modules.convert_to_n5 import convert_to_n5
#mod = convert_to_n5.Convert2N5()
#mod.run()

#downsample n5
#from acpreprocessing.stitching_modules.downsample_n5 import downsample_n5
#mod = downsample_n5.DownsampleN5()
#mod.run()

#test parsemetadata
#from acpreprocessing.stitching_modules.metadata  import parse_metadata
#mod = parse_metadata.ParseMetadata()
#print(mod.get_md())

#test multiscale
#from acpreprocessing.stitching_modules.multiscale_viewing  import multiscale
#mod = multiscale.Multiscale()
##mod.run()

state = {"layers": []}

from acpreprocessing.stitching_modules.nglink import create_layer, create_nglink
for pos in range (3):
    demo_input = {
        "position": pos,
        "outputDir": "/ACdata/processed/demoModules/output/"
    }
    mod1 = create_layer.NgLayer(input_json = demo_input)
    mod1.run(state)
    

mod2 = create_nglink.Nglink()
mod2.run(state)

from acpreprocessing.stitching_modules.stitch import create_json
mod = create_json.CreateJson()
outputDir = "/ACdata/processed/demoModules/output/"
mod.run(outputDir)

from acpreprocessing.stitching_modules.stitch import stitch
mod = stitch.Stitch()
outputDir = "/ACdata/processed/demoModules/output/"
stitchjson = outputDir + "stitch.json"
mod.run(stitchjson)


print('done testing')
print(time.time()-start)
