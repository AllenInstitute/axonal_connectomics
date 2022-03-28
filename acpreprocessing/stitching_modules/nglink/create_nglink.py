from acpreprocessing.stitching_modules.nglink import write_nglink
from argschema.fields import Str
import argschema
import os
from acpreprocessing.utils import io

example_input = {
    "outputDir": "/ACdata/processed/demoModules/output/",
    "fname": "nglink.txt"
}


class CreateNglinkSchema(argschema.ArgSchema):
    outputDir = Str(required=True, description='output directory')
    fname = Str(default="nglink.txt", description='output filename for nglink')


class Nglink(argschema.ArgSchemaParser):
    default_schema = CreateNglinkSchema

    def run(self, state):
        encoded_url = write_nglink.make_neuroglancer_url_vneurodata(state)
        write_nglink.write_url(self.args['outputDir'], self.args['fname'],encoded_url)
        # save state (overrite)
        f_out = os.path.join(self.args['outputDir'], 'state.json')
        io.save_metadata(f_out, state)


if __name__ == '__main__':
    mod = Nglink()
    state = {"layers": []}
    mod.run(state)
