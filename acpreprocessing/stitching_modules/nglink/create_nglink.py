from acpreprocessing.stitching_modules.nglink import write_nglink

from argschema import ArgSchemaParser, ArgSchema
from argschema.fields import NumpyArray, Boolean,Float, Int, Str
import argschema

example_input = {
    "outputDir": "/ACdata/processed/testModules/"
}

class CreateNglinkSchema(argschema.ArgSchema):
    outputDir = argschema.fields.String(default='', description='output root directory')

class Nglink():
    def run(self, state):
        mod = ArgSchemaParser(input_data=example_input,schema_type=CreateNglinkSchema)
        write_nglink.write_url(mod.args['outputDir'], state)

if __name__ == '__main__':
    mod = Nglink()
    state = {"layers": []}
    mod.run(state)