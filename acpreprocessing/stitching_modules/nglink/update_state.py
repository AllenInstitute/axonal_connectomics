from acpreprocessing.stitching_modules.metadata import parse_metadata
from acpreprocessing.stitching_modules.nglink import create_nglink
from argschema.fields import Int, Str
import argschema
from acpreprocessing.utils import io
import os

example_input = {
    'rootDir': "/ACdata/processed/demoModules/raw/",
    "outputDir": "/ACdata/processed/demoModules/output/",
    'mip_level': 3,
    "fname": "stitched-nglink.txt"
    }


class UpdateStateSchema(argschema.ArgSchema):
    rootDir = Str(required=True, description='raw tiff root directory')
    outputDir = Str(default='', description='output directory')
    mip_level = Int(required=False, default=2,
                    description='downsample level to perform stitching with')
    fname = Str(default="nglink.txt", description='output filename for nglink')


# update statejson with stitched coordinates
def update_positions(statejson, stitchoutjson, n_pos, factor):
    for pos in range(0, n_pos):
        # print(pos)
        try:
            statejson['layers'][pos]['source']['transform']['matrix'][0][3] = stitchoutjson[pos]['position'][0]*factor
            statejson['layers'][pos]['source']['transform']['matrix'][1][3] = stitchoutjson[pos]['position'][1]*factor
            statejson['layers'][pos]['source']['transform']['matrix'][2][3] = stitchoutjson[pos]['position'][2]*factor
        except IndexError:
            print("Something went wrong with the stitching output!")


class UpdateState(argschema.ArgSchemaParser):
    default_schema = UpdateStateSchema

    def run(self):
        md_input = {
            "rootDir": self.args['rootDir']
        }
        n_pos = parse_metadata.ParseMetadata(input_data=md_input).get_number_of_positions()
        statejson = io.read_json(os.path.join(self.args['outputDir'],
                                              "state.json"))
        stitchoutjson = io.read_json(os.path.join(self.args['outputDir'],
                                                  "stitch-final.json"))
        update_positions(statejson, stitchoutjson,
                         n_pos, 2**self.args['mip_level'])

        # create stitched nglink
        nglink_input = {
            "outputDir": self.args['outputDir'],
            "fname": "stitched-nglink.txt"
        }
        if not os.path.exists(os.path.join(self.args['outputDir'],"stitched-nglink.txt")):
            create_nglink.Nglink(input_data=nglink_input).run(statejson)
        else:
            print("stitched-nglink.txt already exists!")
        

        io.save_metadata(os.path.join(self.args['outputDir'],
                                      "stitched-state.json"), statejson)


if __name__ == '__main__':
    mod = UpdateState(example_input)
    mod.run()
