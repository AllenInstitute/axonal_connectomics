#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  5 13:35:27 2021

Description: Crops and saves all tiff files in specified directory. Assumes cropping only in x direction. 
Input arguments: -o      output filename prefix
                 -left   x coordinate of top left corner of crop box
                 -right  x coordinate of bottom right corner of crop box
                 -i      input directory path (string)
@author: shubha.bhaskaran
"""
import json
from PIL import Image, ImageSequence, TiffImagePlugin
import sys
import os

def main():
     cwd = os.getcwd()
     args = sys.argv[1:]
     if len(args) == 8 and args[0] == '-o' and args[2] == '-left' and args[4] == '-right' and args[6] == "-i":
         fo = args[1]
         left = int(args[3])
         right = int(args[5])
         input_dir = args[7]
         direct = os.listdir(input_dir)
         direct.sort()
         k=0
         for tiff in direct:
             if tiff.endswith(".tif"):
                 print(tiff)
                 img = Image.open(tiff)
                 box = (left, 0, right,img.size[1])
                 pages = []
                 for i, page in enumerate(ImageSequence.Iterator(img)):
                     with TiffImagePlugin.AppendingTiffWriter(fo+str(k)+".tif") as tf:
                         cropped = page.crop(box)
                         meta_dict = {str(key) : page.tag[key] for key in page.tag.keys()}
                         # print(meta_dict)
                         res = json.loads(meta_dict['51123'][0])
                         #index;file;pixelResolution;type
                         inf = str(i)+";"+tiff+";"+str(res["PixelSizeUm"])+";"+str(res["PixelType"])
                         head = {270:(inf)}
                         cropped.save(tf, tiffinfo=head)
                         tf.newFrame()
                 k=k+1
                 
     else:
         sys.exit("Input Argument Error")

if __name__ == "__main__":
    main()