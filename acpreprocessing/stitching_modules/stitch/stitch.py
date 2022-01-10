import os
from argschema import ArgSchemaParser, ArgSchema
from argschema.fields import Str
from acpreprocessing.stitching_modules.stitch import create_json
import subprocess

def stitch(stitchjson):
    curdir = os.getcwd()
    #os.chdir('/ACdata/stitching-spark/startup-scripts/')
    subprocess.call(["python", "/ACdata/stitching-spark/startup-scripts/spark-local/stitch.py","-i",stitchjson])
    #os.system('python spark-local/stitch.py -i %s'%(stitchjson))
    os.chdir(curdir)
    print("Finished stitching")

class StitchSchema(ArgSchema):
    stitchjson = Str(required=True, description='inputjson for stitching')

class Stitch(ArgSchemaParser):
    default_schema = StitchSchema
    
    def run(self):
        stitch(self.args['stitchjson'])

if __name__ == '__main__':
    outputDir = "/ACdata/processed/demoModules/output/"
    stitchjson = outputDir + "stitch.json"
    Stitch.run(stitchjson)
