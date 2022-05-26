

import argschema
import os
from acpreprocessing.utils import io, convert
import numpy as np
from scipy.ndimage import gaussian_filter
import scipy.ndimage as ndimage
from argschema.fields import Str, Float
import matplotlib.pyplot as plt

#Adapted from Jun Wang's code

example_input = {
    "input_filename": "/Users/sharmishtaas/Documents/data/axonal/M6data/Section_13/ex1_2_13.tif",
    "flatten_method": "top",
    "output_filename": "test.tif",
    "flip_back": False
}


def ndfilter(img,sig=3):
    img = ndimage.gaussian_filter(img, sigma=(sig, sig, 5), order=0)
    return img

def flatten_bottom(img, bottom):
    """
    shift the height of each pixel to align the bottom of the section
    :param img: 3d array
    :param bottom: 2d array, int, indices of bottom surface
    :return imgb: 3d img, same size as img,
    """

    if len(img.shape) != 3:
        raise ValueError('input array should be 3d.')

    if bottom.shape != (img.shape[1], img.shape[2]):
        raise ValueError('the shape of top should be the same size as each plane in img.')

    imgb = np.zeros(img.shape, dtype=img.dtype)

    z, y, x = img.shape

    for yi in range(y):
        for xi in range(x):
            b = bottom[yi, xi]
            if b!= 0:
                col = img[:b, yi, xi]
                imgb[-len(col):, yi, xi] = col

    imgb = imgb[-np.amax(bottom):, :, :]

    return imgb


def flatten_top(img, top):
    """
    shift the height of each pixel to align the top of the section
    :param img: 3d array
    :param top: 2d array, int, indices of top surface
    :return imgft: 3d img, same size as img,
    """

    if len(img.shape) != 3:
        raise ValueError('input array should be 3d.')

    if top.shape != (img.shape[1], img.shape[2]):
        raise ValueError('the shape of top should be the same size as each plane in img.')

    imgt = np.zeros(img.shape, dtype=img.dtype)

    z, y, x = img.shape

    for yi in range(y):
        for xi in range(x):
            t = top[yi, xi]
            col = img[t:, yi, xi]
            imgt[:len(col), yi, xi] = col

    #imgt = imgt[-(z-np.amin(top)):, :, :]
    return imgt

def up_crossings(data, threshold=0):
    """
    find the index where the data up cross the threshold. return the indices of all up crossings (the onset data point
    that is greater than threshold, 1d-array). The input data should be 1d array.
    """
    if len(data.shape) != 1:
        raise ValueError('Input data should be 1-d array.')

    pos = data > threshold
    return (~pos[:-1] & pos[1:]).nonzero()[0] + 1

def find_surface(img, surface_thr, top_buffer = 0, bot_buffer = 30, is_plot=False):
    """
    :param img: 3d array, ZYX, assume small z = top; large z = bottom
    :param surface_thr: [0, 1], threshold for detecting surface
    :return top: 2d array, same size as each plane in img, z index of top surface
    :return bot: 2d array, same size as each plane in img, z index of bottom surface
    """

    if len(img.shape) != 3:
        raise ValueError('input array should be 3d.')

    z, y, x = img.shape

    top = np.zeros((y, x), dtype=np.int)
    bot = np.ones((y, x), dtype=np.int) * z

    if is_plot:
        f = plt.figure(figsize=(5, 5))
        ax = f.add_subplot(111)

    for yi in range(y):
        for xi in range(x):
            curr_t = img[:, yi, xi]
            mx = curr_t.max()
            mn = curr_t.min()
            if mx != mn:
                curr_t = (curr_t - mn) / (mx - mn)

                if is_plot:
                    if yi % 10 == 0 and xi % 10 == 0:
                        ax.plot(range(len(curr_t)), curr_t, '-b', lw=0.5, alpha=0.1)

                if curr_t[0] < surface_thr:
                    curr_top = up_crossings(curr_t, surface_thr)
                    cur_top = cur_top + top_buffer
                    if len(curr_top) != 0:
                        top[yi, xi] = curr_top[0]

                if curr_t[-1] < surface_thr:
                    curr_bot = down_crossings(curr_t, surface_thr)
                    curr_bot = curr_bot+bot_buffer
                    if len(curr_bot) != 0:
                        bot[yi, xi] = curr_bot[-1]

    if is_plot:
        plt.show()

    return top, bot

def down_crossings(data, threshold=0):
    """
    find the index where the data down cross the threshold. return the indices of all down crossings (the onset data
    point that is less than threshold, 1d-array). The input data should be 1d array.
    """
    if len(data.shape) != 1:
        raise ValueError('Input data should be 1-d array.')

    pos = data < threshold
    return (~pos[:-1] & pos[1:]).nonzero()[0] + 1


def flatten_both_sides(img, top, bottom):
    """
    flatten both sides by interpolation
    :param img: 3d array
    :param top: 2d array, int, indices of top surface
    :param bottom: 2d array, int, indices of bottom surface
    :return imgtb: 3d img
    """

    if len(img.shape) != 3:
        raise ValueError('input array should be 3d.')

    if bottom.shape != (img.shape[1], img.shape[2]):
        raise ValueError('the shape of top should be the same size as each plane in img.')

    if top.shape != (img.shape[1], img.shape[2]):
        raise ValueError('the shape of top should be the same size as each plane in img.')

    z, y, x = img.shape

    depths = bottom - top
    depth = int(np.median(depths.flat))
    

    imgtb = np.zeros((depth, y, x), dtype=img.dtype)

    colz_tb = np.arange(depth)

    for yi in range(y):
        for xi in range(x):
            col = img[top[yi, xi]:bottom[yi, xi], yi, xi]
            colz = np.arange(len(col))
            imgtb[:, yi, xi] = np.interp(x=colz_tb, xp=colz, fp=col)

    return imgtb


class FlattenSchema(argschema.ArgSchema):
    input_filename = Str(required=True, description='Input File')
    flatten_method = Str(required=True, validator=marshmallow.validate.OneOf(["top", "bottom"]), description='Type of flattening (top, bottom, both)')
    output_filename = Str(required=True, description='Output File')
    threshold = Float(default=0.3, description = 'Threshold for finding surface')
    

class Flatten(argschema.ArgSchemaParser):
    default_schema = FlattenSchema

    
    def run(self):
        thresh = self.args['threshold']
        I = io.get_tiff_image(self.args['input_filename'])
        IM = convert.downsample_stack_volume(I, dsfactors = (2,4,4))
        I_flip =  np.rot90(IM,1,(0,2))
        I_flip_smoothed = ndfilter(I_flip)

        top,bottom = find_surface(I_flip_smoothed, thresh, is_plot=False)

        if self.args['flatten_method'] == 'top':
            I_flat = flatten_top(I_flip,top)
        elif self.args['flatten_method'] == 'bottom':
            I_flat = flatten_bottom(I_flip,bottom)
        else:
            print("Please choose correct flattening method: top or bottom")
            sys.exit()

        if self.args['flip_back']:
                I_flat = np.rot90(I_flat,3,(0,2))
        io.save_tiff_image(I_flat, self.args['output_filename'])
        
        

    

    

if __name__ == '__main__':
    mod = Flatten(example_input)
    
    mod.run()
