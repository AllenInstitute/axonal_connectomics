from acpreprocessing.stitching_modules.metadata  import parse_metadata

from argschema import ArgSchemaParser, ArgSchema
from argschema.fields import NumpyArray, Boolean,Float, Int, Str
import argschema

example_input = {
    "outputDir": "/ACdata/processed/demoModules/output/",
    "position": 2
}

def create_layer(outputDir, position, overlap, pixelResolution):
    layer_info = {"type": "image"}
    layer_info["shaderControls"] = { "normalized": { "range": [ 0, 5000 ] }}
    url = "n5://http://bigkahuna.corp.alleninstitute.org"
    url = url + outputDir + 'Pos%d/multirespos%d'%(position, position)
    layer_info["source"] = {"url": url}
    layer_info["name"] = "Pos%d"%(position)
    layer_info["source"]["transform"] = {
        "matrix" : [
            [1, 0, 0, 0],
            [0, 1, 0, overlap*position],
            [0, 0, 1, 0]
        ],
        "outputDimensions" : {"x":[pixelResolution[0],"um"], "y":[pixelResolution[1],"um"], "z":[pixelResolution[2],"um"]}
    }
    return layer_info

def add_layer(state, layer_info):
	state["layers"].append(layer_info)


class CreateLayerSchema(argschema.ArgSchema):
    position = argschema.fields.Int(default=0, description='acquisition strip position number')
    outputDir = argschema.fields.String(default='', description='output directory')

class NgLayer():
    def run(self, state):
        mod = ArgSchemaParser(input_data=example_input,schema_type=CreateLayerSchema)
        md = parse_metadata.ParseMetadata()
        pr = md.get_pixel_resolution()
        overlap = md.get_overlap()
        layer0 = create_layer(mod.args['outputDir']+'n5/', mod.args['position'],overlap, pr)
        add_layer(state, layer0)

if __name__ == '__main__':
    mod = NgLayer()
    state = {"layers": []}
    mod.run(state)