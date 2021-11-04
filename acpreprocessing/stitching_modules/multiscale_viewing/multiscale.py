import os
from argschema import ArgSchemaParser, ArgSchema
from argschema.fields import NumpyArray, Boolean, Int, Str
import numpy as np
import argschema

class MultiscaleSchema(argschema.ArgSchema):
    Position = argschema.fields.Int(default=0, description='acquisition strip position number')
    output_root = argschema.fields.String(default='', description='output root directory')
    pixelResolution = NumpyArray(dtype=np.float, required=True,description='Pixel Resolution in um')

def add_multiscale_attributes(output_root, pixelResolution,Position):
    curdir = os.getcwd()
    os.chdir(output_root)
    attributes = open("attributes.json", "w")
    attributes.write('{"pixelResolution" : {"unit":"um","dimensions":[%f,%f,%f]},"scales":[[1,1,1],[2,2,2],[4,4,4],[8,8,8],[16,16,16],[32,32,32],[64,64,64],[128,128,128]]}'%(pixelResolution[0], pixelResolution[1], pixelResolution[2]))
    attributes.close()
    os.system('cp attributes.json %s/Pos%d/multirespos%d'%(output_root, Position, Position))
    os.chdir(output_root + 'Pos%d/multirespos%d'%(Position, Position))
    os.system('ln -s ../pos%d s0'%(Position))
    os.chdir(output_root + 'Pos%d/pos%d'%(Position, Position))
    os.system('ln -s ../pos%d pos%d'%(Position, Position))
    os.chdir(curdir)

if __name__ == '__main__':

    # this defines a default dictionary that will be used if input_json is not specified
    example_input = {
            "output_root": "/ACdata/processed/testnglink/n5/",
            "Position": 0,
            "pixelResolution": [0.26, 0.26, 1]

    }
    # here is my ArgSchemaParser that processes my inputs
    mod = ArgSchemaParser(input_data=example_input,
                          schema_type=MultiscaleSchema)

    add_multiscale_attributes(mod.args['output_root'], mod.args['pixelResolution'], mod.args['Position'])
