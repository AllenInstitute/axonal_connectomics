import os
import argschema
from argschema.fields import NumpyArray, Int, Str
from acpreprocessing.utils.metadata import parse_metadata
from acpreprocessing.utils import io

example_input = {
    "position": 0,
    "outputDir": "/ACdata/processed/rmt_testKevin/ispim2/n5/",
    "max_mip": 4,
    "rootDir": "/ACdata/processed/demoModules/raw/",
    "fname": 'acqinfo_metadata.json'
    }


class MultiscaleSchema(argschema.ArgSchema):
    position = Int(required=True,
                   description='acquisition strip position number')
    outputDir = Str(required=True, description='output directory')
    max_mip = Int(default=4, description='Number of downsamples to perform')
    rootDir = Str(required=True, description='raw tiff root directory')
    fname = Str(required=False, default='acqinfo_metadata.json',
                description='name of metadata json file')


def fix_version(outputRoot):
    """fix n5 version in attributes json (overwrites)"""
    att = {"n5": "2.5.0"}
    io.save_metadata(outputRoot+f"/attributes.json", att)


def add_downsampling_factors(outputRoot, pos, max_mip):
    """add downsampling factor key/value to attributes json"""
    for mip_level in range(1, max_mip+1):
        factor = [2**mip_level, 2**mip_level, 2**mip_level]
        d = {"downsamplingFactors": factor}
        att = io.read_json(outputRoot +
                           f"/multirespos{pos}/s{mip_level}/attributes.json")
        att.update(d)
        io.save_metadata(outputRoot +
                         f"/multirespos{pos}/s{mip_level}/attributes.json",
                         att)


def add_multiscale_attributes(outputRoot, pixelResolution, position, max_mip):
    """add attribute file to mutirespos folders and create symlinks"""
    multires_att = os.path.join(outputRoot +
                                f"/multirespos{position}/attributes.json")
    if not os.path.isfile(multires_att):
        attr = {"pixelResolution": {"unit": "um", "dimensions":
                                    [pixelResolution[0],
                                     pixelResolution[1],
                                     pixelResolution[2]]},
                "scales": [[1, 1, 1]]}
        for m in range(1, max_mip+1):
            attr["scales"].append([2**m, 2**m, 2**m])

        io.save_metadata(multires_att, attr)

    # TODO: Is there a way to do symlinks with absolute paths?
    curdir = os.getcwd()
    os.chdir(outputRoot + f"/multirespos{position}")
    if not os.path.islink("s0"):
        os.system(f"ln -s ../pos{position} s0")
    os.chdir(outputRoot + f"/pos{position}")
    if not os.path.islink(f"pos{position}"):
        os.system(f"ln -s ../pos{position} pos{position}")
    os.chdir(curdir)

    add_downsampling_factors(outputRoot, position, max_mip)
    fix_version(outputRoot)


class Multiscale(argschema.ArgSchemaParser):
    default_schema = MultiscaleSchema

    def run(self):
        md_input = {
                "rootDir": self.args['rootDir'],
                "fname": self.args['fname']
                }
        pr = parse_metadata.ParseMetadata(input_data=md_input).get_pixel_resolution()

        add_multiscale_attributes(self.args['outputDir'], pr,
                                  self.args['position'], self.args['max_mip'])
        print("Finished multiscale conversion for Pos%d" %
              (self.args['position']))


if __name__ == '__main__':
    mod = Multiscale()
    mod.run()
