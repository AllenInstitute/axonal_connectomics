import argschema
from argschema import ArgSchemaParser, ArgSchema
from argschema.fields import Str
import json
import numpy as np

example_input = {
    "rootDir": "/m2_data/iSPIM1/test/"
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
        
if __name__ == '__main__':
    mod = ParseMetadata()
    print(mod.get_md())