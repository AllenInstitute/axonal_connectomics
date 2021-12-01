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
    position = Int(required=True, description='acquisition strip position number')
    rootDir = Str(required=True, description='raw tiff root directory')
    outputDir = Str(required=True, description='output directory')
    dsName = Str(default='ex1', description='dataset name')

class Nglink():
    def __init__(self, input_json=example_input):
        self.input_data = input_json.copy()

    def run(self, state, fname):
        mod = ArgSchemaParser(input_data=self.input_data,schema_type=CreateNglinkSchema)
        write_nglink.write_url(mod.args['outputDir'], state, fname)

if __name__ == '__main__':
    mod = Nglink()
    state = {"layers": []}
    mod.run(state)