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


def fix_version(outputRoot):
    att = {"n5":"2.5.0"}
    io.save_metadata(outputRoot+f"/attributes.json",att)

def add_downsampling_factors(outputRoot,position, max_mip):
    for mip_level in range(1,max_mip+1):
        factor = [2**mip_level,2**mip_level,2**mip_level]
        d = {"downsamplingFactors":factor}
        att = io.get_json(outputRoot+f"/multirespos{position}/s{mip_level}/attributes.json")
        att.update(d)
        io.save_metadata(outputRoot+f"/multirespos{position}/s{mip_level}/attributes.json",att)

def add_multiscale_attributes(outputRoot,pixelResolution,position,max_mip):
    #TODO: check if files exist
    if max_mip == 4: #default
        attr = {"pixelResolution" : {"unit":"um","dimensions":[pixelResolution[0],pixelResolution[1],pixelResolution[2]]},"scales":[[1,1,1],[2,2,2],[4,4,4],[8,8,8],[16,16,16]]}
    else:
        attr = {"pixelResolution" : {"unit":"um","dimensions":[pixelResolution[0],pixelResolution[1],pixelResolution[2]]},"scales":[[1,1,1]]}
        for m in range(1,max_mip+1):
            attr["scales"].append([2**m,2**m,2**m])

    multires_att = os.path.join(outputRoot+f"/multirespos{position}/attributes.json")
    io.save_metadata(multires_att,attr)

    #TODO: figure out how to do symlinks with absolute paths`
    curdir = os.getcwd()
    os.chdir(outputRoot + f"/multirespos{position}")
    os.system(f"ln -s ../pos{position} s0")
    os.chdir(outputRoot + f"/pos{position}")
    os.system(f"ln -s ../pos{position} pos{position}")
    os.chdir(curdir)

    add_downsampling_factors(outputRoot,position,max_mip)
    fix_version(outputRoot)

class Multiscale(argschema.ArgSchemaParser):
    default_schema = MultiscaleSchema

    def run(self):
        md_input = {
                "rootDir": "/ispim2_data/MN7_RH_3_b5_S16_1_high_res_region",
                "fname":'acqinfo_metadata.json'
                }
        md = parse_metadata.ParseMetadata(input_data = md_input)
        md.run()
        pr = md.get_pixel_resolution()
        add_multiscale_attributes(self.args['outputDir'], pr, self.args['position'],self.args['max_mip'])
        print("Finished multiscale conversion for Pos%d"%(self.args['position']))

if __name__ == '__main__':
    mod = Multiscale()
    mod.run()
    
