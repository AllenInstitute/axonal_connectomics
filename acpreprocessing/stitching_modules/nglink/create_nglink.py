from acpreprocessing.stitching_modules.nglink import create_state, write_nglink
from acpreprocessing.stitching_modules.metadata  import parse_metadata

from argschema import ArgSchemaParser, ArgSchema
from argschema.fields import NumpyArray, Boolean,Float, Int, Str
import argschema

example_input = {
    "outputRoot": "/ACdata/processed/testModules/",
    "position": 0
}

class CreateNglinkSchema(argschema.ArgSchema):
    position = argschema.fields.Int(default=0, description='acquisition strip position number')
    outputRoot = argschema.fields.String(default='', description='output root directory')

class Nglink():
    def run(self):
        mod = ArgSchemaParser(input_data=example_input,schema_type=CreateNglinkSchema)
        state = {"layers": []}
        md = parse_metadata.ParseMetadata()
        pr = md.get_pixel_resolution()
        overlap = md.get_overlap()
        layer0 = create_state.create_layer(mod.args['outputRoot']+'n5/', mod.args['position'],overlap, pr)
        create_state.add_layer(state, layer0)
        write_nglink.write_url(mod.args['outputRoot'], state)

if __name__ == '__main__':
    mod = Nglink()
    mod.run()