from acpreprocessing.stitching_modules.metadata import parse_metadata
from argschema import ArgSchemaParser, ArgSchema
from argschema.fields import NumpyArray, Boolean,Float, Int, Str
import json
import argschema
from acpreprocessing.utils import io
import os

example_input = {
    "rootDir": "/ACdata/processed/demoModules/raw/",
    "outputDir": "/ACdata/processed/demoModules/output/",
}

def get_pos_info(downdir, overlap, pos, pr, ind):
    att = io.get_json(downdir+"attributes.json")
    sz = att["dimensions"]
    yshift = overlap/4
    if att['dataType']=='uint16':
        dtype = 'GRAY16'
    pos_info= {"file":downdir,"index":ind,"pixelResolution":pr,"position":[0,ind*yshift,0],"size":sz,"type":dtype}
    return pos_info

class CreateJsonSchema(argschema.ArgSchema):
    rootDir = Str(required=True, description='raw tiff root directory')
    outputDir = Str(required=True, description='output directory')

class CreateJson(argschema.ArgSchemaParser):
    default_schema = CreateJsonSchema

    def run(self, n_start, n_end):
        stitching_json = []
        md_input = {
                'rootDir':'/ispim2_data/MN7_RH_3_b5_S16_1_high_res_region'
                }
        md = parse_metadata.ParseMetadata(input_data =md_input)
        # n_pos = md.get_number_of_positions()
        ind=0
        for pos in range(n_start,n_end):
            downdir = self.args['outputDir']+f"/Pos{pos}.n5/multirespos{pos}/s2/"
            pos_info = get_pos_info(downdir, md.get_overlap(), pos, md.get_pixel_resolution(),ind)
            stitching_json.append(pos_info)
            ind = ind+1
        fout = os.path.join(self.args['outputDir'],'stitch.json')
        io.save_metadata(fout,stitching_json)
        
if __name__ == '__main__':
    mod = CreateJson()
    mod.run()
