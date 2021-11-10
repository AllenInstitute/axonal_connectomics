import os
from argschema import ArgSchemaParser, ArgSchema
from argschema.fields import NumpyArray, Boolean,Float, Int, Str

example_input = {
    "position": 0,
    "outputDir": "/ACdata/processed/testModules/"
}

def slice_tiff_to_n5(outputDir,position):
    curdir = os.getcwd()
    os.chdir('/allen/programs/celltypes/workgroups/em-connectomics/analysis_group/forSharmi/axonal/n5-spark/startup-scripts/')
    os.system('python spark-local/slice-tiff-to-n5.py -i %s -n %s -o pos%d -b 64,64,64 -c RAW'%(outputDir+'2Dtiff/',outputDir+'n5/',position))
    os.chdir(curdir)
    print("Finished n5 conversion for Pos%d"%(position))

class Convert2N5Schema(ArgSchema):
    position = Int(required=True, description='acquisition strip position number')
    outputDir = Str(required=True, description='output directory') 

class Convert2N5():
    def run(self):
        mod = ArgSchemaParser(input_data=example_input,schema_type=Convert2N5Schema)
        slice_tiff_to_n5(mod.args['outputDir'],mod.args['position'])


if __name__ == '__main__':
    mod = Convert2N5()
    mod.run()