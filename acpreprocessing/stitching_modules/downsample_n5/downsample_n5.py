import os
from argschema import ArgSchemaParser, ArgSchema
from argschema.fields import NumpyArray, Boolean,Float, Int, Str

example_input = {
    "position": 0,
    "outputDir": "/ACdata/processed/testModules/"
}

def downsample_n5(outputDir,position):
    curdir = os.getcwd()
    os.chdir('/allen/programs/celltypes/workgroups/em-connectomics/analysis_group/forSharmi/axonal/n5-spark/startup-scripts/')
    os.system('python spark-local/n5-scale-pyramid.py -n %s -i pos%d -f 2,2,2 -o multirespos%d'%(outputDir+'n5/',position,position))
    os.chdir(curdir)
    

class DownsampleN5Schema(ArgSchema):
    position = Int(required=True, description='acquisition strip position number')
    outputDir = Str(required=True, description='output directory')

class DownsampleN5():
    def run(self):
        mod = ArgSchemaParser(input_data=example_input,schema_type=DownsampleN5Schema)
        downsample_n5(mod.args['outputDir'],mod.args['position'])

if __name__ == '__main__':
    mod = Convert2N5()
    mod.run()