import argschema
from argschema import ArgSchemaParser, ArgSchema
from argschema.fields import Str
import json
import numpy as np
from acpreprocessing.utils import io

example_input = {
    "rootDir": "/ACdata/processed/demoModules/raw/",
    "fname":'acqinfo_metadata.json'
}

class ParseMetadataSchema(argschema.ArgSchema):
    rootDir = Str(required=True, description='raw tiff root directory')
    fname = Str(required=True, description='name of metadata json')

class ParseMetadata():
    def __init__(self,input_json=example_input):
        self.input_data = input_json.copy()
        mod = ArgSchemaParser(input_data=self.input_data,schema_type=ParseMetadataSchema)
        self.rootDir = mod.args["rootDir"]
        self.md = io.get_json(self.rootDir+mod.args["fname"])

    #Return metadata json
    def get_md(self):
        return self.md
    
    #return pixel resolution in um
    def get_pixel_resolution(self):
        xy = self.md['settings']['pixel_spacing_um']
        z = self.md['positions'][1]['x_step_um']
        return [xy,xy,z]
    
    #return overlap in pixels
    def get_overlap(self):
        return self.md['settings']['strip_overlap_pixels']
    
    #TODO: get from metadata file once updated
    def get_number_of_positions(self):
       return 10
    
    # #TODO:
    # def get_size(self):
    #     sz = [self.md['settings']['image_size_xy'][0],self.md['settings']['image_size_xy'][1],403*3]
    #     return sz

    #get data type
    def get_dtype(self):
        return self.md['settings']['dtype']

if __name__ == '__main__':
    mod = ParseMetadata()
    print(mod.get_md())
