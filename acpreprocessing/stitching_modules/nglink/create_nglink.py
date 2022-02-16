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
        write_nglink.write_tinyurl(self.args['outputDir'],
                                   state, self.args['fname'])
        # save state (overrite)
        f_out = os.path.join(self.args['outputDir'], 'state.json')
        io.save_metadata(f_out, state)


if __name__ == '__main__':
    mod = Nglink()
    state = {"layers": []}
    mod.run(state)
