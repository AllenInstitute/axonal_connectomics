import os
import shutil
from argschema import ArgSchemaParser, ArgSchema
from argschema.fields import NumpyArray, Int, Str
import numpy as np
import argschema
from acpreprocessing.stitching_modules.metadata  import parse_metadata
from acpreprocessing.utils import io

example_input = {
    "position": 0,
    "outputDir": "/ACdata/processed/rmt_testKevin/ispim2/n5/",
    "max_mip": 4
    }

class MultiscaleSchema(argschema.ArgSchema):
    position = Int(required=True, description='acquisition strip position number')
    outputDir = Str(required=True, description='output directory')
    max_mip = Int(default=4, description='Number of downsamples')

def add_multiscale_attributes(outputRoot, pixelResolution,position):
    #TODO: check if files exist
    attr = {"pixelResolution" : {"unit":"um","dimensions":[pixelResolution[0],pixelResolution[1],pixelResolution[2]]},"scales":[[1,1,1],[2,2,2],[4,4,4],[8,8,8],[16,16,16]]}
    highest_att = os.path.join(outputRoot, "attributes.json")
    if not os.path.exists(highest_att):
        io.save_metadata(os.path.join(highest_att,attr))
    shutil.copy(highest_att,outputRoot+f"Pos{position}.n5/multirespos{position}")
    #TODO: figure out how to do symlinks with absolute paths`
    curdir = os.getcwd()
    os.chdir(outputRoot + f"Pos{position}.n5/multirespos{position}")
    os.system(f"ln -s ../pos{position} s0")
    os.chdir(outputRoot + f"Pos{position}.n5/pos{position}")
    os.system(f"ln -s ../pos{position} pos{position}")
    os.chdir(curdir)

def add_downsampling_factors(outputRoot,position, max_mip):
    factor = [1,1,1]
    d = {"downsamplingFactors":factor}
    att = io.get_json(outputRoot+f"Pos{position}.n5/pos{position}/attributes.json")
    att.update(d)
    io.save_metadata(outputRoot+f"Pos{position}.n5/pos{position}/attributes.json",att)
    for mip_level in range(1,max_mip+1):
        factor = [2**mip_level,2**mip_level,2**mip_level]
        d = {"downsamplingFactors":factor}
        att = io.get_json(outputRoot+f"Pos{position}.n5/multirespos{position}/s{mip_level}/attributes.json")
        att.update(d)
        io.save_metadata(outputRoot+f"Pos{position}.n5/multirespos{position}/s{mip_level}/attributes.json",att)

class Multiscale(argschema.ArgSchemaParser):
    default_schema = MultiscaleSchema

    def run(self):
        md = parse_metadata.ParseMetadata()
        pr = md.get_pixel_resolution()
        add_multiscale_attributes(self.args['outputDir'], pr, self.args['position'])
        add_downsampling_factors(self.args['outputDir'],self.args['position'],self.args['max_mip'])
        print("Finished multiscale conversion for Pos%d"%(self.args['position']))

if __name__ == '__main__':
    mod = Multiscale()
    mod.run()
    
