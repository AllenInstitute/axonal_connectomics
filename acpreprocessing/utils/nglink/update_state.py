import posix
from acpreprocessing.utils.metadata import parse_metadata
from acpreprocessing.utils.nglink import create_nglink
from argschema.fields import Int, Str, Boolean
import argschema
from acpreprocessing.utils import io
import os

example_input = {
    'rootDir': "/ACdata/processed/demoModules/raw/",
    "outputDir": "/ACdata/processed/demoModules/output/",
    'mip_level': 3,
    "fname": "stitched-nglink.txt",
    "consolidate_pos": True
    }


class UpdateStateSchema(argschema.ArgSchema):
    rootDir = Str(required=True, description='raw tiff root directory')
    outputDir = Str(default='', description='output directory')
    mip_level = Int(required=False, default=2,
                    description='downsample level to perform stitching with')
    fname = Str(default="nglink.txt", description='output filename for nglink')
    consolidate_pos = Boolean(required=False, default=True,
                              description="Whether to consolidate all position"
                                          "strips in one neuroglancer layer or"
                                          "separate them into independent layers")
    n_channels = Int(required=True,description="total number of channels in acquisition")
    state_json = Str(required=False,default="state.json", description="Name of overview state json file")
    stitch_final = Str(required=False,default="stitch-final.json", description="Name of final stitching result json file")
    stitched_state = Str(required=False,default="stitched_state.json", description="Name of stitched state json file")
    stitched_nglink = Str(required=False,default="stitched-nglink.txt", description="Name of stitched nglink txt file")


# update statejson with stitched coordinates
def update_positions(statejson, stitchoutjson, n_pos, factor, n_channels):
    for channel in range(0, n_channels):
        for pos in range(0, n_pos):
            try:
                statejson['layers'][pos+channel*n_pos]['source'][0]['transform']['matrix'][0][3] = stitchoutjson[pos]['position'][0]*factor
                statejson['layers'][pos+channel*n_pos]['source'][0]['transform']['matrix'][1][3] = stitchoutjson[pos]['position'][1]*factor
                statejson['layers'][pos+channel*n_pos]['source'][0]['transform']['matrix'][2][3] = stitchoutjson[pos]['position'][2]*factor
            except IndexError:
                print("Something went wrong with the stitching output!")
                # print(pos)


def update_positions_consolidated(statejson, stitchoutjson, n_pos, factor, n_channels):
    for channel in range(0, n_channels):
        for pos in range(0, n_pos):
            # print(pos)
            try:
                statejson['layers'][channel]['source'][pos]['transform']['matrix'][0][3] = stitchoutjson[pos]['position'][0]*factor
                statejson['layers'][channel]['source'][pos]['transform']['matrix'][1][3] = stitchoutjson[pos]['position'][1]*factor
                statejson['layers'][channel]['source'][pos]['transform']['matrix'][2][3] = stitchoutjson[pos]['position'][2]*factor
            except IndexError:
                print("Something went wrong with the stitching output!")
                print(pos)


class UpdateState(argschema.ArgSchemaParser):
    default_schema = UpdateStateSchema

    def run(self):
        md_input = {
            "rootDir": self.args['rootDir']
        }
        n_pos = parse_metadata.ParseMetadata(input_data=md_input).get_number_of_positions()
        statejson = io.read_json(os.path.join(self.args['outputDir'],
                                              self.args["state_json"]))
        stitchoutjson = io.read_json(os.path.join(self.args['outputDir'],
                                                  self.args["stitch_final"]))
        if self.args["consolidate_pos"]:
            update_positions_consolidated(statejson, stitchoutjson,
                                          n_pos, 2**self.args['mip_level'], self.args["n_channels"])
        else:
            update_positions(statejson, stitchoutjson,
                             n_pos, 2**self.args['mip_level'], self.args["n_channels"])

        # create stitched nglink
        nglink_input = {
            "outputDir": self.args['outputDir'],
            "fname": self.args["stitched_nglink"]
        }
        if not os.path.exists(os.path.join(nglink_input['outputDir'], nglink_input["fname"])):
            create_nglink.Nglink(input_data=nglink_input).run(statejson)
        else:
            print(f"{nglink_input['fname']} already exists")

        io.save_metadata(os.path.join(self.args['outputDir'],
                                      self.args["stitched_state"]), statejson)


if __name__ == '__main__':
    mod = UpdateState(example_input)
    mod.run()
