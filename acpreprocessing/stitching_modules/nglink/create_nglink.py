from acpreprocessing.stitching_modules.nglink import create_state, write_nglink

from argschema import ArgSchemaParser, ArgSchema
from argschema.fields import NumpyArray, Boolean,Float, Int, Str
import numpy as np
import argschema

example_input = {
    "outputRoot": "/ACdata/processed/testModules/ngLink/n5/",
    "position": 0,
    "pixelResolution": [0.26, 0.26, 1],
    "overlap": 509.53846153846166,
    "dsName":"testnglink"
}

class CreateNglinkSchema(argschema.ArgSchema):
    position = argschema.fields.Int(default=0, description='acquisition strip position number')
    outputRoot = argschema.fields.String(default='', description='output root directory')
    pixelResolution = NumpyArray(dtype=float, required=True,description='Pixel Resolution in um')
    overlap = argschema.fields.Float(required=True, description='overlap between position strips')
    dsName = argschema.fields.String(default='', description='dataset name')

class Nglink():
    def run(self):
        mod = ArgSchemaParser(input_data=example_input,schema_type=CreateNglinkSchema)
        state = {"layers": []}

        layer0 = create_state.create_layer(mod.args['outputRoot'], mod.args['position'],mod.args['overlap'], mod.args['pixelResolution'])
        create_state.add_layer(state, layer0)
        write_nglink.write_url(mod.args['dsName'], state)


if __name__ == '__main__':
    mod = Nglink()
    mod.run()