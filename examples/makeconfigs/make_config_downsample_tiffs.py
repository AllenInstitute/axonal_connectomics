import glob
import os
import json

def create_metadata_filename(tokens, output_directory):
	return os.path.join(output_directory, tokens[-2], tokens[-1].replace("tif", "json"))
def create_downsampled_filename(tokens,output_directory):
	return os.path.join(output_directory, tokens[-2],tokens[-1])

input_directory = "/ispim1_data/487748_48_NeuN_NFH_488_25X_0.5XPBS"
output_image_directory = "/ispim1_data/processed/487748_48_NeuN_NFH_488_25X_0.5XPBS/downsampled_tifs"
output_metadata_directory = "/ispim1_data/processed/487748_48_NeuN_NFH_488_25X_0.5XPBS/metadata"

files = glob.glob("%s/global_l40*/*.tif"%input_directory)
alldicts = []

for f in files:
	tokens = f.split("/")
	info = dict()
	info['raw_file'] = f
	info['metadata_file'] = create_metadata_filename(tokens,output_metadata_directory)
	info['downsampled_file'] = create_downsampled_filename(tokens,output_image_directory)

	alldicts.append(info)

with open("configs/config_downsample_tifs_487748_48_NeuN_NFH_488_25X_0.5XPBS.json", 'w') as file:
                json_string = json.dumps(alldicts, default=lambda o: o.__dict__, sort_keys=True, indent=2)
                file.write(json_string)



