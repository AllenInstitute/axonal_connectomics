import os
from argschema import ArgSchemaParser, ArgSchema
from argschema.fields import NumpyArray, Int, Str
import numpy as np
import argschema

example_input = {
    "outputRoot": "/ACdata/processed/testnglink/n5/",
    "position": 0,
    "pixelResolution": [0.26, 0.26, 1]
}

class MultiscaleSchema(argschema.ArgSchema):
    position = argschema.fields.Int(default=0, description='acquisition strip position number')
    outputRoot = argschema.fields.String(default='', description='output root directory')
    pixelResolution = NumpyArray(dtype=float, required=True,description='Pixel Resolution in um')

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
        add_multiscale_attributes(mod.args['outputRoot'], mod.args['pixelResolution'], mod.args['position'])

if __name__ == '__main__':
    mod = Multiscale()
    mod.run()
    
