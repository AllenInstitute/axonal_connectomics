from argschema.fields import Int, Str, Boolean
from argschema.fields import Int, Str
from acpreprocessing.utils.metadata import parse_metadata
from argschema.fields import Int, Str, Boolean
import argschema
from acpreprocessing.utils import io
import os
import posixpath

example_input = {
    "rootDir": "/ACdata/processed/demoModules/raw/",
    "outputDir": "/ACdata/processed/demoModules/output/",
    "mip_level": 2,
    "reverse": False
}


class CreateJsonSchema(argschema.ArgSchema):
    rootDir = Str(required=True, description='raw tiff root directory')
    outputDir = Str(required=True, description='output directory')
    mip_level = Int(required=False, default=2,
                    description='downsample level to perform stitching with')
    reverse = Boolean(required=False, description='Whether position strips should be placed in reverse order')
    dirname = Str(required = True, description="name of dataset folder")

# Create specific position strip information needed for stitching
# (including approximate coordinates using overlap)
def get_pos_info(downdir, overlap, pr, ind, mip_level, reverse):
    factor = 2**mip_level
    att = io.read_json(downdir+"attributes.json")
    sz = att["dimensions"]
    yshift = sz[0]-overlap/factor
    if att['dataType'] == 'uint16':
        dtype = 'GRAY16'
    if reverse:
        pos_info = {"file": downdir, "index": ind, "pixelResolution": pr,
                    "position": [0, -1*ind*yshift, 0], "size": sz, "type": dtype}
    else:
        pos_info = {"file": downdir, "index": ind, "pixelResolution": pr,
                    "position": [0, ind*yshift, 0], "size": sz, "type": dtype}
    return pos_info


class CreateJson(argschema.ArgSchemaParser):
    default_schema = CreateJsonSchema

    def run(self):
        stitching_json = []
        md_input = {
                'rootDir': self.args['rootDir']
                }
        md = parse_metadata.ParseMetadata(input_data=md_input)
        n_pos = md.get_number_of_positions()

        for pos in range(0, n_pos):
            # downdir = self.args['outputDir'] + f"/Pos{pos}.n5/multirespos{pos}/s2/"
            downdir = posixpath.join(
                    self.args["outputDir"]+self.args["dirname"]+".n5/",
                    f"setup{pos}/timepoint0/s{self.args['mip_level']}/")
            # print(downdir)
            pos_info = get_pos_info(downdir, md.get_overlap(),
                                    md.get_pixel_resolution(), pos,
                                    self.args['mip_level'],self.args["reverse"])
            stitching_json.append(pos_info)

        fout = os.path.join(self.args['outputDir'], 'stitch.json')
        io.save_metadata(fout, stitching_json)


if __name__ == '__main__':
    CreateJson(example_input).run()
