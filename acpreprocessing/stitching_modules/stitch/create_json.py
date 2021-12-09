from acpreprocessing.stitching_modules.metadata import parse_metadata
from argschema import ArgSchemaParser, ArgSchema
from argschema.fields import NumpyArray, Boolean,Float, Int, Str
import json
import argschema
from acpreprocessing.utils import io

example_input = {
    "position": 2,
    "rootDir": "/ACdata/processed/demoModules/raw/",
    "outputDir": "/ACdata/processed/demoModules/output/",
    'dsName':'ex1'
}

def get_pos_info(downdir, overlap, pos, pr, ind):
    att = io.get_json(downdir+"attributes.json")
    sz = att["dimensions"]
    yshift = overlap/4
    if att['dataType']=='uint16':
        dtype = 'GRAY16'
    pos_info= {"file":downdir,"index":ind,"pixelResolution":pr,"position":[0,pos*yshift,0],"size":sz,"type":dtype}
    return pos_info

class CreateJsonSchema(argschema.ArgSchema):
    position = Int(required=True, description='acquisition strip position number')
    rootDir = Str(required=True, description='raw tiff root directory')
    outputDir = Str(required=True, description='output directory')
    dsName = Str(default='ex1', description='dataset name')

class CreateJson():
    def __init__(self, input_json=example_input):
        self.input_data = input_json.copy()
        
    def run(self, n_start, n_end):
        mod = ArgSchemaParser(input_data=self.input_data,schema_type=CreateJsonSchema)
        stitching_json = []
        md = parse_metadata.ParseMetadata()
        # n_pos = md.get_number_of_positions()
        ind=0
        for pos in range(n_start,n_end):
            downdir = mod.args['outputDir']+"/n5/Pos%d/multirespos%d/s2/"%(pos, pos)
            pos_info = get_pos_info(downdir, md.get_overlap(), pos, md.get_pixel_resolution(),ind)
            stitching_json.append(pos_info)
            ind = ind+1
        io.save_metadata(mod.args['outputDir']+'stitch.json',stitching_json)
        
if __name__ == '__main__':
    mod = CreateJson()
    outputDir = "/ACdata/processed/demoModules/output/"
    mod.run(outputDir)
