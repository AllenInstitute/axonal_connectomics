# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import json
from PIL import Image, ImageSequence
import sys
import os
from utils import io


def main():
     cwd = os.getcwd()
     args = sys.argv[1:]
     if len(args) == 2 and args[0] == '-o':
         fo = args[1]
         data = []
         k = 0
         direct = os.listdir(cwd)
         direct.sort()
         for tiff in direct:
             if tiff.endswith(".tif") and tiff.startswith("cropmore"):
                 print("processing "+tiff)
                 img = Image.open(tiff)
                 si = img.size
                 meta_dict = {str(key) : img.tag[key] for key in img.tag.keys()}
                 # print(meta_dict)
                 inf = meta_dict['270'][0].split(';')
                 head = {"index":[],"file":[],"position":[],"size":[],"pixelResolution":[],"type":[]}
                 head['index'] = k
                 head['file'] = cwd+"/"+tiff
                 #change this when we get more information
                 head['position'] = [0,0-(400*k),0]
                 head['size'] = [si[0],si[1],img.n_frames]
                 #if cropped tiff
                 head['pixelResolution'] = [float(inf[2]),float(inf[2]),float(inf[2])]
                 head['type'] = inf[3]
                 #if uncropped original tiff
                 data.append(head)
                 k=k+1
         io.save_metadata(fo+".json", data)
         
     else:
         sys.exit("Input Argument Error")

if __name__ == "__main__":
    main()