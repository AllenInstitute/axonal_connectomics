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

    def get_md(self):
        """Return entire metadata json"""
        return self.md

    def get_pixel_resolution(self):
        """Return x,y,z pixel resolution in um"""
        xy = self.md['settings']['pixel_spacing_um']
        z = self.md['positions'][1]['x_step_um']
        return [xy, xy, z]

    def get_overlap(self):
        """Return strip overlap in pixels - needed for stitching input"""
        return self.md['settings']['strip_overlap_pixels']

    def get_number_of_positions(self):
        """Return number of position strips in section"""
        return len(self.md['positions'])

    def get_number_of_channels(self):
        """Return number of channels of data for section"""
        return self.md['channels']

    def get_direction(self):
        """Get direction of y direction for position strips in section"""
        if (self.md["positions"][0]["y_start_um"]) > (self.md["positions"][1]["y_start_um"]):
            return True
        else:
            return False
    
    def get_size(self):
        """Get tiff width, height, and number of frames in stack"""
        sz = [self.md['settings']['image_size_xy'][0],
              self.md['settings']['image_size_xy'][1],
              self.md['settings']['frames_per_file']]
        return sz

    def get_dtype(self):
        """Get data type (uint8, uint16, ...)"""
        return self.md['settings']['dtype']

    def get_angle(self):
        """Get imaging sheet angle - needed for deskewing"""
        return self.md['settings']['sheet_angle']


if __name__ == '__main__':
    mod = ParseMetadata(input_data=example_parsemetadata_input)
    print(mod.get_md())
