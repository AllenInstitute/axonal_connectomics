import create_state
import write_nglink

from argschema import ArgSchemaParser, ArgSchema
from argschema.fields import NumpyArray, Boolean,Float, Int, Str
import numpy as np
import argschema

class createNglinkSchema(argschema.ArgSchema):
    Position = argschema.fields.Int(default=0, description='acquisition strip position number')
    output_root = argschema.fields.String(default='', description='output root directory')
    pixelResolution = NumpyArray(dtype=np.float, required=True,description='Pixel Resolution in um')
    overlap = argschema.fields.Float(required=True, description='overlap between position strips')
    ds_name = argschema.fields.String(default='', description='dataset name')

if __name__ == '__main__':

    # this defines a default dictionary that will be used if input_json is not specified
    example_input = {
            "output_root": "/ACdata/processed/testnglink/n5/",
            "Position": 0,
            "pixelResolution": [0.26, 0.26, 1],
            "overlap": 509.53846153846166,
            'ds_name':'testnglink'
    }
    # here is my ArgSchemaParser that processes my inputs
    mod = ArgSchemaParser(input_data=example_input,
                          schema_type=createNglinkSchema)

    state = {"layers": []}

    layer0 = create_state.create_layer(mod.args['output_root'], mod.args['Position'],mod.args['overlap'], mod.args['pixelResolution'])
    create_state.add_layer(state, layer0)
    write_nglink.write_url(mod.args['ds_name'], state)
