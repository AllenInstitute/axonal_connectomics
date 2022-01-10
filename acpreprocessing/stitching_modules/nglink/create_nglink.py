from acpreprocessing.stitching_modules.nglink import write_nglink
from argschema import ArgSchemaParser, ArgSchema
from argschema.fields import NumpyArray, Boolean,Float, Int, Str
import argschema
import os
from acpreprocessing.utils import io

example_input = {
    "fname": "nglink.txt",
    "outputDir": "/ACdata/processed/demoModules/output/",
}

class CreateNglinkSchema(argschema.ArgSchema):
    outputDir = Str(required=True, description='output directory')
    fname = Str(default="nglink.txt", description='output filename for nglink')

class Nglink(argschema.ArgSchemaParser):
    default_schema = CreateNglinkSchema
    
    def run(self, state):
        write_nglink.write_tinyurl(self.args['outputDir'], state, self.args['fname'])
        #save state
        f_out = os.path.join(self.args['outputDir'],'state.json')
        io.save_metadata(f_out,state)

if __name__ == '__main__':
    mod = Nglink()
    state = {"layers": []}
    mod.run(state)
