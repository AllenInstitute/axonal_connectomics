import os
from argschema import ArgSchemaParser, ArgSchema
from argschema.fields import NumpyArray, Boolean,Float, Int, Str

example_input = {
    "position": 2,
    "outputDir": "/ACdata/processed/demoModules/output/"
}

def downsample_n5(outputDir,position):
    curdir = os.getcwd()
    os.chdir('/allen/programs/celltypes/workgroups/em-connectomics/analysis_group/forSharmi/axonal/n5-spark/startup-scripts/')
    os.system('python spark-local/n5-scale-pyramid.py -n %s -i pos%d -f 2,2,2 -o multirespos%d'%(outputDir,position,position))
    os.chdir(curdir)

class DownsampleN5Schema(ArgSchema):
    position = Int(required=True, description='acquisition strip position number')
    outputDir = Str(required=True, description='output directory')

class DownsampleN5():
    def run(self):
        mod = ArgSchemaParser(input_data=example_input,schema_type=DownsampleN5Schema)
        downsample_n5(mod.args['outputDir']+"n5/Pos%d/"%(mod.args['position']),mod.args['position'])
        print("Finished n5 downsample")
        
if __name__ == '__main__':
    mod = DownsampleN5()
    mod.run()