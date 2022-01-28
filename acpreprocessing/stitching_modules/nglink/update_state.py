from acpreprocessing.stitching_modules.metadata import parse_metadata
from argschema import ArgSchemaParser, ArgSchema
from argschema.fields import NumpyArray, Boolean, Float, Int, Str
import argschema
from acpreprocessing.utils import io
import os

example_input = {
    'rootDir': "/ACdata/processed/demoModules/raw/",
    "outputDir": "/ACdata/processed/demoModules/output/",
    'mip_level': 3
    }


class UpdateStateSchema(argschema.ArgSchema):
    rootDir = Str(required=True, description='raw tiff root directory')
    outputDir = argschema.fields.String(default='',
                                        description='output directory')
    mip_level = Int(required=False, default=2,
                    description='downsample level to perform stitching with')


class UpdateState(argschema.ArgSchemaParser):
    default_schema = UpdateStateSchema

    def run(self):
        md_input = {
            "rootDir": self.args['rootDir']
        }
        md = parse_metadata.ParseMetadata(input_data=md_input)
        n_pos = md.get_number_of_positions()
        statejson = io.read_json(os.path.join(self.args['outputDir'], "state.json"))
        stitchoutjson = io.read_json(os.path.join(self.args['outputDir'], "stitch-final.json"))
        factor = 2**self.args['mip_level']
        for pos in range(0, n_pos):
            try:
                statejson['layers'][pos]['source']['transform']['matrix'][0][3] = stitchoutjson[pos]['position'][0]*factor
                statejson['layers'][pos]['source']['transform']['matrix'][1][3] = stitchoutjson[pos]['position'][1]*factor
                statejson['layers'][pos]['source']['transform']['matrix'][2][3] = stitchoutjson[pos]['position'][2]*factor
            except IndexError:
                print("Something went wrong with the stitching output!")


if __name__ == '__main__':
    mod = UpdateState(example_input)
    mod.run()
