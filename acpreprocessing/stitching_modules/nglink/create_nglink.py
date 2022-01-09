from acpreprocessing.stitching_modules.nglink import write_nglink

from argschema import ArgSchemaParser, ArgSchema
from argschema.fields import NumpyArray, Boolean,Float, Int, Str
import argschema

example_input = {
    "position": 2,
    "rootDir": "/ACdata/processed/demoModules/raw/",
    "outputDir": "/ACdata/processed/demoModules/output/",
    'dsName':'ex1'
}

class CreateNglinkSchema(argschema.ArgSchema):
    outputDir = Str(required=True, description='output directory')
    fname = Str(default="nglink.txt", description='output filename for nglink')

class Nglink(argschema.ArgSchemaParser):
    default_schema = CreateNglinkSchema
    def run(self, state):
        #write_nglink.write_url(self.args['outputDir'], state, self.args['fname'])
        write_nglink.write_tinyurl(self.args['outputDir'], state, self.args['fname'])

if __name__ == '__main__':
    mod = Nglink()
    state = {"layers": []}
    mod.run(state)
