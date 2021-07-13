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

def get_pixel_overlap(direct, input_dir):
    #get w adjacent tiff stacks' headers
    for file in direct:
        if file.endswith(".tif"):
            prefix = os.path.splitext(os.path.splitext(file)[0])[0][0:-1]
            break
    files = [input_dir + prefix + "1.ome.tif",input_dir + prefix + "2.ome.tif"]
    it = 0
    for f in files:
        img = Image.open(f)
        if it == 0:
            md0 = {str(key) : img.tag[key] for key in img.tag.keys()}
            header0 = json.loads(md0['51123'][0])
        if it == 1:
            md1 = {str(key) : img.tag[key] for key in img.tag.keys()}
            header1 = json.loads(md1['51123'][0])
        it = it+1
        
    #Calculate the pixel overlap
    diff = abs(float(header1['XYStage:XY:31-ScanSlowAxisStopPosition(mm)'])-float(header0['XYStage:XY:31-ScanSlowAxisStopPosition(mm)']))
    FOV = header1['PixelSizeUm']*header1["Height"]
    pixeloverlap = (FOV-(diff*1000))/header1['PixelSizeUm']
    print("PixelOverlap: "+ pixeloverlap)
    return pixeloverlap
    
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
         pixeloverlap = get_pixel_overlap(direct, input_dir)
         
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
                 head['position'] = [0,0+(pixeloverlap*k),0]
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