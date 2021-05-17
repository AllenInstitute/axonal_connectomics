#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description: Parses all tiff file headers in specified directory. Outputs json file containing specific stitching information.
Input arguments: -crop   whether tiff files are cropped or not (boolean)
                 -i      input directory path (string
                 -o      output filename prefix
@author: shubha.bhaskaran
"""

import json
from PIL import Image
import sys
import os
from utils import io


def parse_cropped(inf, head):
    head['pixelResolution'] = [float(inf[2]),float(inf[2]),float(inf[2])]
    head['type'] = inf[3]
    
def parse_uncropped(res, head):
    head['pixelResolution'] = [float(res['PixelSizeUm']),float(res['PixelSizeUm']),float(res['PixelSizeUm'])]
    head['type'] = res['PixelType']
    

def main():
     args = sys.argv[1:]
     if len(args) == 6 and args[0] == '-crop' and args[2] == '-i' and args[4] == '-o':
         input_dir = args[3]
         fo = args[5]
         cropped = args[1]
         data = []
         k = 0
         direct = os.listdir(input_dir)
         direct.sort()
         #TODO fix magic numebrs
         #FOV in um
         FOV = 2048*6.5/25
         pixeloverlap = 510
         for file in direct:
             if file.endswith(".tif"):
                 print("parsing "+ file)
                 img = Image.open(input_dir + file)
                 si = img.size
                 meta_dict = {str(key) : img.tag[key] for key in img.tag.keys()}
                 # print(meta_dict)
                 head = {"index":[],"file":[],"position":[],"size":[],"pixelResolution":[],"type":[]}
                 head['index'] = k
                 #TODO: fix filepath to be on s3 instead of local if running on emr
                 head['file'] = input_dir+"/"+file
                 #TODO change this when we get more information
                 head['position'] = [0,0-(pixeloverlap*k),0]
                 head['size'] = [si[0],si[1],img.n_frames]
                 #if cropped tiff
                 if cropped == "True":
                     parse_cropped(meta_dict['270'][0].split(';'),head)
                 else:
                     parse_uncropped(json.loads(meta_dict['51123'][0]), head)
                 data.append(head)
                 k=k+1
         io.save_metadata(input_dir+fo+".json", data)
         
     else:
         sys.exit("Input Argument Error")

if __name__ == "__main__":
    main()