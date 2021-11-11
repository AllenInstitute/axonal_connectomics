import os
from argschema import ArgSchemaParser, ArgSchema
from argschema.fields import NumpyArray, Int, Str
import numpy as np
import argschema
from acpreprocessing.stitching_modules.metadata  import parse_metadata

example_input = {
    "outputRoot": "/ACdata/processed/testModules/",
    "rootDir": "/m2_data/iSPIM1/test/",
    "position": 0
}

class MultiscaleSchema(argschema.ArgSchema):
    position = argschema.fields.Int(equired=True, description='acquisition strip position number')
    outputRoot = argschema.fields.String(equired=True, description='output root directory')
    rootDir = Str(required=True, description='raw tiff root directory')

def add_multiscale_attributes(outputRoot, pixelResolution,position):
    curdir = os.getcwd()
    os.chdir(outputRoot)
    attributes = open("attributes.json", "w")
    attributes.write('{"pixelResolution" : {"unit":"um","dimensions":[%f,%f,%f]},"scales":[[1,1,1],[2,2,2],[4,4,4],[8,8,8],[16,16,16],[32,32,32],[64,64,64],[128,128,128]]}'%(pixelResolution[0], pixelResolution[1], pixelResolution[2]))
    attributes.close()
    os.system('cp attributes.json %s/Pos%d/multirespos%d'%(outputRoot, position, position))
    os.chdir(outputRoot + 'Pos%d/multirespos%d'%(position, position))
    os.system('ln -s ../pos%d s0'%(position))
    os.chdir(outputRoot + 'Pos%d/pos%d'%(position, position))
    os.system('ln -s ../pos%d pos%d'%(position, position))
    os.chdir(curdir)

class Multiscale():
    def run(self):
        mod = ArgSchemaParser(input_data=example_input,schema_type=MultiscaleSchema)
        md = parse_metadata.ParseMetadata()
        pr = md.get_pixel_resolution()
        add_multiscale_attributes(mod.args['outputRoot']+'n5/', pr, mod.args['position'])
        print("Finished multiscale conversion for Pos%d"%(mod.args['position']))

if __name__ == '__main__':
    mod = Multiscale()
    mod.run()
    
