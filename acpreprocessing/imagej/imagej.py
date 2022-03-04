import copy
from multiprocessing.sharedctypes import Value
import os
import subprocess

class ImageJRunner:
    def __init__(self,
                 imagej_path=None,
                 config={},
                 script_type='macro',
                 script_path=None,
                 args={}):
        if imagej_path is None:
            self.imagej_path = os.getenv(
                'IMAGEJ_PATH',
                default='Fiji.app/ImageJ-linux64'
            )
        else:
            self.imagej_path = imagej_path
        
        if 'script_type' in config:
            self.script_type = config['script_type']
        else:
            self.script_type = script_type
        
        if 'script_path' in config:
            self.script_path = config['script_path']
        else:
            self.script_path = script_path
        
        if 'args' in config:
            self.args = copy.deepcopy(config['args'])
        else:
            self.args = copy.deepcopy(args)
        
        self.argstring = self._generate_argstring()
        self.imagej_cmd = self._generate_cmd()
        
        if os.path.isfile(self.imagej_path):
            raise ValueError("ERROR: ImageJ not found")

    def print_script(self):
        with open(self.imagej_path) as fp:
            lines = fp.readlines()
        
        for line in lines:
            print(line)

    def run(self):
        if len(self.imagej_cmd) > 0:
            subprocess.run(self.imagej_cmd)
        else:
            raise ValueError('')
    
    def _generate_argstring(self):
        arglist = []
        for arg in self.args:
            arglist.append(self.args[arg])

            if self.args[arg].endswith('_dir'):
                os.makedirs(self.args[arg], exist_ok=True)
        
        argdelim = "#"
        argstring = argdelim.join(arglist)

        return argstring

    def _generate_cmd(self):
        imagej_cmd = [self.imagej_path,
                        '--headless',
                        '-macro', os.path.join(self.script_path, self.script),
                        self.argstring]
        
        return imagej_cmd