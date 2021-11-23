from acpreprocessing.stitching_modules.metadata import parse_metadata
from argschema import ArgSchemaParser, ArgSchema
from argschema.fields import NumpyArray, Boolean,Float, Int, Str
import json

class CreateJson():
    def run(self, outputDir):
        stitching_json = []
        md = parse_metadata.ParseMetadata()
        n_pos = md.get_number_of_positions()
        pr = md.get_pixel_resolution()
        overlap = md.get_overlap()
        sz = [576,576,254]
        #sz = md.get_size()
        #sz2 = [int(sz[0]/4),int(sz[1]/4),int(sz[2]/4)]
        yshift = (overlap/2304)*576
        #dtype = md.get_dtype()
        dtype = 'GRAY16'
        for pos in range(n_pos):
            pos_info= {"file":outputDir+"/n5/Pos%d/multirespos%d/s2/"%(pos, pos),"index":pos,"pixelResolution":pr,"position":[0,pos*yshift,0],"size":sz,"type":dtype}
            stitching_json.append(pos_info)
        with open(outputDir+'stitch.json', 'w') as outfile:
            json.dump(stitching_json, outfile)
        
if __name__ == '__main__':
    mod = CreateJson()
    outputDir = "/ACdata/processed/demoModules/output/"
    mod.run(outputDir)
