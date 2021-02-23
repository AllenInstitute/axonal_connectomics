from .utils import io
from .utils import convert
import numpy as np
import os

def downsample_tiffs_and_extract_metadata(config):
	
	print(config)

	input_filename = config['raw_file']
	output_image_file = config['downsampled_file']
	output_metadata_file = config['metadata_file']
	print("These are the files: ")
	print(input_filename)
	print(output_image_file)

	if not os.path.exists(output_image_file):

		if not os.path.exists(output_image_file.rsplit('/',1)[0]):
			os.makedirs(output_image_file.rsplit('/',1)[0])

		if not os.path.exists(output_metadata_file.rsplit('/',1)[0]):
                	os.makedirs(output_metadata_file.rsplit('/',1)[0])

		m = io.get_metadata(input_filename)
		io.save_metadata(output_metadata_file,m)

		I = io.get_tiff_image(input_filename)
		I_ds = convert.downsample_stack(I,4)
		I_ds_c = convert.clip_and_adjust_range_values(I_ds)
		I_ds_c_16 = np.asarray(I_ds_c, dtype=np.int16)
		io.save_tiff_image(I_ds_c_16,output_image_file)
