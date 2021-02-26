from .utils import io
from .utils import convert
import numpy as np
import os
from PIL import Image
import time

def downsample_tiff_and_extract_metadata(config):
	
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
		print("Saved metadata")
		I = io.get_tiff_image(input_filename)
		print("reading tif")
		I_ds = convert.downsample_stack(I,4)
		print("downsampled")
		I_ds_c = convert.clip_and_adjust_range_values(I_ds)
		print("clip and adjust")
		I_ds_c_16 = np.asarray(I_ds_c, dtype=np.int16)
		print("convert type")
		io.save_tiff_image(I_ds_c_16,output_image_file)
		print("saved")


def downsample_gif(config):
        
	print(config)

	input_filename = config['raw_file']
	output_image_file = config['downsampled_file']

	print("These are the files: ")
	print(input_filename)
	print(output_image_file)
	iftimer = True

	if not os.path.exists(output_image_file):
		if not os.path.exists(output_image_file.rsplit('/',1)[0]):
			os.makedirs(output_image_file.rsplit('/',1)[0])
		time0=time.time()
		#print(time0)
		I = io.get_tiff_image(input_filename)
		if iftimer: print ("Read file: ", time.time() - time0)
		I_ds = convert.downsample_stack(I,4)
		if iftimer: print ("downsampled ", time.time() - time0)
		I_ds_c = convert.clip_and_adjust_range_values(I_ds)
		if iftimer: print ("converted to 8 bit ", time.time() - time0) 
		imgs = [Image.fromarray(img) for img in I_ds_c]
		imgs[0].save(fpath, save_all=True, append_images=imgs[1:], duration=25, loop=0)
		if iftimer: print ("wrote gif ", time.time() - time0)