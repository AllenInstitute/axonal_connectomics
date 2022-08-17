from acpreprocessing.utils.nglink import create_layer, create_nglink
from argschema.fields import Dict
import argschema
import os

example_input = {
    "run_input": {
        "outputDir": "/ACdata/processed/demoModules/output/",
        "rootDir": "/ACdata/processed/demoModules/raw/",
        "ds_name": MN6_2_S83_220531_high_res,
        "mip_level": 3,
        "md_filename": "/ACdata/processed/demoModules/raw/acqinfo_metadata.json",
        "consolidate_pos": True,
        "reverse_stitch": False,
        "deskew": False
        }
    }


class CreateOverviewSchema(argschema.ArgSchema):
    run_input = argschema.fields.Dict(required=True, description='Input json for processing')

class Overview(argschema.ArgSchemaParser):
    default_schema = CreateOverviewSchema

    def run(self, n_channels, n_pos, dirname, deskew, state=None):
        if not state:
            state = {"showDefaultAnnotations": False, "layers": []}
        for channel in range(n_channels):
            for pos in range(n_pos):
                if self.run["run_input"]["consolidate_pos"]:
                    layer_input = {
                            "position": 0,
                            "outputDir": self.run["run_input"]['outputDir']+dirname+".n5/channel"+str(channel),
                            "rootDir": self.run["run_input"]['rootDir'],
                            "reverse": self.run["run_input"]["reverse_stitch"],
                            "deskew": deskew,
                            "channel": channel
                            }
                    create_layer.NgLayer(input_data=layer_input).run_consolidate(state)
                    break
                else:
                    layer_input = {
                        "position": pos,
                        "outputDir": self.run["run_input"]['outputDir']+dirname+".n5/channel"+str(channel),
                        "rootDir": self.run["run_input"]['rootDir'],
                        "reverse": self.run["run_input"]["reverse_stitch"],
                        "deskew": deskew,
                        "channel": channel
                        }
                    create_layer.NgLayer(input_data=layer_input).run(state)

        # Create nglink from created state and estimated positions (overview link)
        nglink_input = {
                "outputDir": self.run["run_input"]['outputDir'],
                "fname": self.run["run_input"]["nglink_name"],
                "state_json": self.run["run_input"]["state_json"]
                }
        if not os.path.exists(os.path.join(nglink_input['outputDir'], nglink_input['fname'])):
            create_nglink.Nglink(input_data=nglink_input).run(state)
        else:
            print("nglink txt file already exists!")


if __name__ == '__main__':
    mod = Overview(input_data=example_input).run()
