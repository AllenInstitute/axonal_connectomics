import json
from PIL import Image
from PIL.TiffTags import TAGS
from pathlib import Path
import tifffile as tf

def get_metadata(filename):
	with Image.open(filename) as img:
		meta_dict = {str(key) : img.tag[key] for key in img.tag.keys()}
		res = json.loads(meta_dict['51123'][0])
		return res

def get_tiff_image(filename):
	data = tf.imread(filename,multifile=False)
	return data

def save_tiff_image(I, filename):
	tf.imsave(filename,I)

def save_metadata(filename,sample):
	with open(filename, 'w') as file:
                json_string = json.dumps(sample, default=lambda o: o.__dict__, sort_keys=True, indent=2)
                file.write(json_string)
