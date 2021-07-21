import os
import subprocess
import time

imagej_path = '/home/samk/Fiji.app/ImageJ-linux64'

hello_input = {
    'script_type': 'macro',
    'script_dir': 'home/samk/axonal_connectomics/imagej/ijm',
    'script': 'hello.ijm',
    'args': {
        'string': 'Hello World',
        'num_prints': 5,
    }
}

example_input = {
    'script_type': 'macro',
    'script_path': '/home/samk/axonal_connectomics/imagej/ijm',
    'script': 'example.ijm',
    'args': {
        'tiff_path': '/ispim1_data/PoojaB/487748_48_NeuN_NFH_488_25X_0.5XPBS/global_l40_Pos001',
        'gif_path': '/ispim1_data/PoojaB/487748_48_NeuN_NFH_488_25X_0.5XPBS/processed/ds_gifs/global_l40_Pos001'
    }
}

class ImageJRunner:
    def __init__(self, imagej_path=imagej_path):
        if os.path.isfile(imagej_path):
            print("ERROR: ImageJ not found")
        
        self.imagej_path = imagej_path

    def load_script(self, script_config):
        self.script_config = script_config
        self.script_type = script_config['script_type']
        self.script_path = script_config['script_path']
        self.script = script_config['script']
        self.args = script_config['args'].copy()

        arglist = []
        for arg in self.args:
            arglist.append(self.args[arg])

            if self.args[arg].endswith('_dir') and \
                not os.path.exists(self.args[arg]):
                os.makedirs(self.args[arg])
        
        argdelim = "#"
        argstring = argdelim.join(arglist)

        self.imagej_cmd = []

        if self.script_type == 'macro':
            imagej_cmd = [self.imagej_path,
                          '--headless',
                          '-macro', os.path.join(self.script_path, self.script),
                          argstring]

    def get_script(self):
        pass

    def print_script(self):
        pass

    def run_script(self):
        if len(self.imagej_cmd) > 0:
            subprocess.run(self.imagej_cmd)
        else:
            print("Must load script first")
    
