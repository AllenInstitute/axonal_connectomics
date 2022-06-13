from acpreprocessing.utils.metadata import parse_metadata
from argschema.fields import Str, Boolean, Int
import argschema

example_input = {
    "outputDir": "/ACdata/processed/demoModules/output/",
    "position": 2,
    "rootDir": "/ACdata/processed/demoModules/raw/",
    "md_filename": "acqinfo_metadata.json"
    }


# Creates an image layer in neuroglancer that points to the n5 data of a specific position
def create_layer(outputDir, position, ypos, pixelResolution, deskew):
    layer_info = {"type": "image"}
    layer_info["shader"] = "#uicontrol invlerp normalized\n#uicontrol float power slider(default=0.5, min=0, max=2)\n\nfloat somepower(float v, float power){\n   return pow(v, power);\n  }\nvoid main() {\n  emitGrayscale(somepower(normalized(), power));\n}\n"
    layer_info["shaderControls"] = {"normalized": {"range": [500, 1500]}}
    url = "n5://http://bigkahuna.corp.alleninstitute.org"
    # os.path.join not working as I thought here?
    url = url + outputDir + '/setup%d/timepoint0/' % (position)
    layer_info["source"] = [{"url": url}]
    layer_info["name"] = "Pos%d" % (position)
    layer_info["source"][position]["transform"] = {
        "matrix": [
            [1, 0, 0, 0],
            [0, 1, 0, ypos*position],
            [deskew, 0, 1, 0]
        ],
        "outputDimensions": {"x": [pixelResolution[0], "um"],
                             "y": [pixelResolution[1], "um"],
                             "z": [pixelResolution[2], "um"]}
    }
    return layer_info


def add_source(outputDir, position, ypos, pixelResolution, state, deskew):
    url = "n5://http://bigkahuna.corp.alleninstitute.org"
    url = url + outputDir + '/setup%d/timepoint0/' % (position)
    state["layers"][0]["source"].append({"url": url})
    state["layers"][0]["source"][position]["transform"] = {
        "matrix": [
            [1, 0, 0, 0],
            [0, 1, 0, ypos*position],
            [deskew, 0, 1, 0]
        ],
        "outputDimensions": {"x": [pixelResolution[0], "um"],
                             "y": [pixelResolution[1], "um"],
                             "z": [pixelResolution[2], "um"]}
    }


# Add layer to state
def add_layer(state, layer_info):
    state["layers"].append(layer_info)


class CreateLayerSchema(argschema.ArgSchema):
    position = argschema.fields.Int(default=0,
                                    description='position strip number')
    outputDir = argschema.fields.String(default='',
                                        description='output directory')
    rootDir = Str(required=True, description='raw tiff root directory')
    md_filename = Str(required=False, default="acqinfo_metadata.json",
                      description='metadata file name')
    reverse = Boolean(required=False,default=False, description="Whether to reverse direction of stitching or not")
    deskew = Int(required=False,default=0, description="deskew factor (0 if want to leave undeskewed)")


class NgLayer(argschema.ArgSchemaParser):
    default_schema = CreateLayerSchema

    def run(self, state=None):
        if state is None:
            state = {"layers": []}
        md_input = {
            "rootDir": self.args['rootDir'],
            "fname": self.args['md_filename']
        }
        md = parse_metadata.ParseMetadata(input_data=md_input)
        pr = md.get_pixel_resolution()
        sz = md.get_size()
        ypos = sz[1]-md.get_overlap()  # subtract height of image by pixel overlap to get yposition

        if self.args["reverse"]:
            layer0 = create_layer(self.args['outputDir'], self.args['position'],
                                  -1*ypos, pr, self.args['deskew'])
        else:
            layer0 = create_layer(self.args['outputDir'], self.args['position'],
                                  ypos, pr, self.args['deskew'])
        add_layer(state, layer0)

    def run_consolidate(self, state=None):
        if state is None:
            state = {"layers": []}
        md_input = {
            "rootDir": self.args['rootDir'],
            "fname": self.args['md_filename']
        }
        md = parse_metadata.ParseMetadata(input_data=md_input)
        pr = md.get_pixel_resolution()
        sz = md.get_size()
        n_pos = md.get_number_of_positions()
        ypos = sz[1]-md.get_overlap()  # subtract height of image by pixel overlap to get yposition
        # Create the layer
        if self.args["reverse"]:
            layer = create_layer(self.args['outputDir'], self.args['position'],
                                 -1*ypos, pr, self.args['deskew'])
        else:
            layer = create_layer(self.args['outputDir'], self.args['position'],
                                 ypos, pr, self.args['deskew'])
        add_layer(state, layer)

        for pos in range(1, n_pos):
            if self.args["reverse"]:
                add_source(self.args['outputDir'], pos, -1*ypos, pr, state, self.args['deskew'])
            else:
                add_source(self.args['outputDir'], pos, ypos, pr, state, self.args['deskew'])




if __name__ == '__main__':
    mod = NgLayer(input_data=example_input).run()
