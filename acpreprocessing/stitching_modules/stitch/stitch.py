import os
from argschema import ArgSchemaParser, ArgSchema
from argschema.fields import NumpyArray, Boolean,Float, Int, Str
from acpreprocessing.stitching_modules.stitch import create_json

def stitch(stitchjson):
    curdir = os.getcwd()
    os.chdir('/ACdata/stitching-spark/startup-scripts/')
    os.system('python spark-local/stitch.py -i %s'%(stitchjson))
    os.chdir(curdir)
    print("Finished stitching")

class Stitch():
    def run(self,stitchjson):
        stitch(stitchjson)

if __name__ == '__main__':
    mod = Stitch()
    outputDir = "/ACdata/processed/demoModules/output/"
    stitchjson = outputDir + "stitch.json"
    mod.run(stitchjson)
