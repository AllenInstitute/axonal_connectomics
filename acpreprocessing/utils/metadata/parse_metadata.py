import argschema
from argschema.fields import Str
from acpreprocessing.utils import io
import os

example_parsemetadata_input = {
    "rootDir": "/ACdata/processed/demoModules/raw/",
    "fname": 'acqinfo_metadata.json'
}


class ParseMetadataSchema(argschema.ArgSchema):
    rootDir = Str(required=True, description='raw tiff root directory')
    fname = Str(required=False, default='acqinfo_metadata.json',
                description='name of metadata json file')


class ParseMetadata(argschema.ArgSchemaParser):

    def __init__(self, input_data=example_parsemetadata_input):
        self.input_data = input_data.copy()
        mod = argschema.ArgSchemaParser(input_data=self.input_data,
                                        schema_type=ParseMetadataSchema)
        self.rootDir = mod.args["rootDir"]
        self.md = io.read_json(os.path.join(self.rootDir, mod.args["fname"]))

    # Return metadata json
    def get_md(self):
        return self.md

    # Return x,y,z pixel resolution in um
    def get_pixel_resolution(self):
        xy = self.md['settings']['pixel_spacing_um']
        z = self.md['positions'][1]['x_step_um']
        return [xy, xy, z]

    # Return strip overlap in pixels
    def get_overlap(self):
        return self.md['settings']['strip_overlap_pixels']

    # Return number of position strips
    def get_number_of_positions(self):
        return len(self.md['positions'])

    # Get tiff stack width, height, and number of frames
    def get_size(self):
        sz = [self.md['settings']['image_size_xy'][0],
              self.md['settings']['image_size_xy'][1],
              self.md['settings']['frames_per_file']]
        return sz

    # Get data type
    def get_dtype(self):
        return self.md['settings']['dtype']

    def get_angle(self):
        return self.md['settings']['sheet_angle']


if __name__ == '__main__':
    mod = ParseMetadata(input_data=example_parsemetadata_input)
    print(mod.get_md())
