"""
Created on Sun Jan 31 19:21:02 2021

@author:  sharmishtaas
"""

import json
# from PIL import Image
# import tifffile as tf
import imageio


def get_metadata(filename):
    with imageio.get_reader(filename, format="actiff") as r:
        res = r.get_meta_data()["MicroManagerMetadata"]
    return res
    # with Image.open(filename) as img:
    #     meta_dict = {str(key) : img.tag[key] for key in img.tag.keys()}
    #     res = json.loads(meta_dict['51123'][0])
    #     return res


def imageio_general_read(*args, read_type=None, **kwargs):
    try:
        return imageio.volread(*args, **kwargs)
    except ValueError:
        try:
            mimgs = imageio.mimread(*args, **kwargs)
            if len(mimgs) == 1 and not read_type == "mimg":
                return mimgs[0]
            else:
                return mimgs
        except ValueError:
            return imageio.imread(*args, **kwargs)


def imageio_general_write(*args, **kwargs):
    try:
        imageio.volwrite(*args, **kwargs)
    except ValueError:
        try:
            imageio.mimwrite(*args, **kwargs)
        except ValueError:
            return imageio.imwrite(*args, **kwargs)


def get_tiff_image(filename):
    data = imageio_general_read(filename, multifile=False, format="actiff")
    # data = tf.imread(filename,multifile=False)
    print(data.shape)
    return data


def save_tiff_image(I, filename):
    imageio_general_write(filename, I, format="actiff")
    # tf.imsave(filename,I)


def save_metadata(filename,sample):
    with open(filename, 'w') as file:
        json_string = json.dumps(sample, default=lambda o: o.__dict__, sort_keys=True, indent=2)
        file.write(json_string)

def read_json(filename):
    with open(filename, 'r') as f:
        data = json.loads(f.read())
        return data

def save_file(filename,sample):
    with open(filename, 'w') as file:
        file.write(sample)

def get_json(filename):
    with open(filename) as f:
        res = json.loads(f.read())
        return res
