from argschema import ArgSchemaParser, ArgSchema
from argschema.fields import Str
import subprocess

example_stitch = {
    "stitchjson": "/ACdata/processed/demoModules/output/stitch.json"
}

def stitch(stitchjson):
    subprocess.call(["python",
                     "/ACdata/stitching-spark/startup-scripts/spark-local/"
                     "stitch.py", "-i", stitchjson])
    print("Finished stitching")


class StitchSchema(ArgSchema):
    stitchjson = Str(required=True, description='inputjson for stitching')


class Stitch(ArgSchemaParser):
    default_schema = StitchSchema

    def run(self):
        stitch(self.args['stitchjson'])


if __name__ == '__main__':
    Stitch.run(input_data=example_stitch)
