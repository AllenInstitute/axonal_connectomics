import os
from argschema import ArgSchemaParser, ArgSchema
from argschema.fields import NumpyArray, Int, Str
import numpy as np
import argschema
from acpreprocessing.stitching_modules.metadata  import parse_metadata
from acpreprocessing.utils import io

example_input = {
    "position": 2,
    "rootDir": "/ACdata/processed/demoModules/raw/",
    "outputDir": "/ACdata/processed/demoModules/output/",
    'dsName':'ex1'
}

class MultiscaleSchema(argschema.ArgSchema):
    position = Int(required=True, description='acquisition strip position number')
    rootDir = Str(required=True, description='raw tiff root directory')
    outputDir = Str(required=True, description='output directory')
    dsName = Str(default='ex1', description='dataset name')

def add_multiscale_attributes(outputRoot, pixelResolution,position):
    curdir = os.getcwd()
    os.chdir(outputRoot)
    attr = {"pixelResolution" : {"unit":"um","dimensions":[pixelResolution[0],pixelResolution[1],pixelResolution[2]]},"scales":[[1,1,1],[2,2,2],[4,4,4],[8,8,8],[16,16,16],[32,32,32],[64,64,64],[128,128,128]]}
    io.save_metadata("attributes.json",attr)
    os.system('cp attributes.json %s/Pos%d/multirespos%d'%(outputRoot, position, position))
    os.chdir(outputRoot + 'Pos%d/multirespos%d'%(position, position))
    os.system('ln -s ../pos%d s0'%(position))
    os.chdir(outputRoot + 'Pos%d/pos%d'%(position, position))
    os.system('ln -s ../pos%d pos%d'%(position, position))
    os.chdir(curdir)

class Multiscale():
    def __init__(self, input_json=example_input):
        self.input_data = input_json.copy()

    def run(self):
        mod = ArgSchemaParser(input_data=self.input_data,schema_type=MultiscaleSchema)
        md = parse_metadata.ParseMetadata()
        pr = md.get_pixel_resolution()
        add_multiscale_attributes(mod.args['outputDir']+'n5/', pr, mod.args['position'])
        print("Finished multiscale conversion for Pos%d"%(mod.args['position']))

if __name__ == '__main__':
    mod = Multiscale()
    mod.run()
    
