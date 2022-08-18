from acpreprocessing.utils.nglink import write_nglink
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
    state_json = Str(required=False,default="state.json", description="Name of overview state json file")


class Nglink(argschema.ArgSchemaParser):
    default_schema = CreateNglinkSchema

    def run(self, state):
        encoded_url = write_nglink.make_neuroglancer_url_vneurodata(state)
        write_nglink.write_url(self.args['outputDir'], self.args['fname'], encoded_url)
        f_out = os.path.join(self.args['outputDir'], self.args['state_json'])
        io.save_metadata(f_out, state)


if __name__ == '__main__':
    mod = Nglink()
    state = {"layers": []}
    mod.run(state)
