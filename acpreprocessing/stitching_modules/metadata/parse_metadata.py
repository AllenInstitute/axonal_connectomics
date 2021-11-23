import argschema
from argschema import ArgSchemaParser, ArgSchema
from argschema.fields import Str
import json
import numpy as np

example_input = {
    "rootDir": "/ACdata/processed/demoModules/raw/"
}

class ParseMetadataSchema(argschema.ArgSchema):
    rootDir = Str(required=True, description='raw tiff root directory')

def read_file(rootDir):
    f = open (rootDir+'acqinfo_metadata.json', "r")
    data = json.loads(f.read())
    return data

class ParseMetadata():
    def __init__(self):  
        self.rootDir = ArgSchemaParser(input_data=example_input,schema_type=ParseMetadataSchema).args['rootDir']
        self.md = read_file(self.rootDir) 
    
    def get_md(self):
        return self.md
    
    def get_pixel_resolution(self):
        xy = self.md['settings']['pixel_spacing_um']
        z = self.md['positions'][1]['x_step_um']
        return [xy,xy,z]
    
    def get_overlap(self):
        return self.md['settings']['strip_overlap_pixels']
    
    #TODO:
    def get_number_of_positions(self):
       return 3
    
    #TODO
    def get_size(self):
        sz = [self.md['settings']['image_size_xy'][0],self.md['settings']['image_size_xy'][1],403*3]
        return sz

    def get_dtype(self):
        return self.md['settings']['dtype']

if __name__ == '__main__':
    mod = ParseMetadata()
    print(mod.get_md())
