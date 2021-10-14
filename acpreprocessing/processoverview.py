#Filename: processoverview.py
#Description: Process raw data and create neuroglancer link of overlapped/stitched positions
#Inputs:
#   0: root directory containing raw ome.tif stacks
#   1: number of positions to process
#   2: number of segments per position to process
#   3: dataset name for output folder
#Returns:
#Last updated: 13th October 2021

from acpreprocessing import downsampling
from acpreprocessing.utils import io,convert
from p2ng import make_neuroglancer_url
import json
import random
import numpy as np
import os
import sys
import time
from operator import methodcaller
import re
from natsort import natsorted, ns
from PIL import Image

def stripfile(filelist, outputdir):
    index = 0
    for inputfile in filelist:
        print(inputfile)
        I = io.get_tiff_image(inputfile)
        for j in range(I.shape[0]):
            img = I[j,:,:]
            fname = outputdir + "/{0:05d}.tif".format(index)
            print(fname)
            io.save_tiff_image(img,fname)
            index +=1

def get_pixel_overlap(input_dir, prefix):
    #get w adjacent tiff stacks' headers
    files = [input_dir + "/"+prefix + "_Pos1.ome.tif",input_dir + "/"+prefix + "_Pos2.ome.tif"]
    img = Image.open(files[0])
    md0 = {str(key) : img.tag[key] for key in img.tag.keys()}
    header0 = json.loads(md0['51123'][0])
    img = Image.open(files[1])
    md1 = {str(key) : img.tag[key] for key in img.tag.keys()}
    header1 = json.loads(md1['51123'][0])

    #Calculate the pixel overlap
    diff = np.subtract(float(header1['XYStage:XY:31-ScanSlowAxisStopPosition(mm)']),float(header0['XYStage:XY:31-ScanSlowAxisStopPosition(mm)']))
    FOV = header1['PixelSizeUm']*header1["Height"]
    pixeloverlap = (FOV-(diff*1000))/header1['PixelSizeUm']
    return header1["Height"]-pixeloverlap

def get_metadata(filename, Position, prefix, overlap):
    img = Image.open(filename)
    si = img.size
    meta_dict = {str(key) : img.tag[key] for key in img.tag.keys()}
    # print(meta_dict)
    res = json.loads(meta_dict['51123'][0])
    head = {"index":[],"file":[],"position":[],"size":[],"pixelResolution":[],"type":[], "stageStepSize":[], "angle":[]}
    head['index'] = Position
    head['file'] = filename
    head['position'] = [0,0+(overlap*Position),0]
    head['size'] = [si[0],si[1],img.n_frames]
    head["stageStepSize"] = abs(float(res["XYStage:XY:31-ScanFastAxisStartPosition(mm)"])-float(res["XYStage:XY:31-ScanFastAxisStopPosition(mm)"]))/float(res["Scanner:AB:33-SPIMNumSlices"])*1000
    head["angle"] = 33 #TODO get from db
    head['pixelResolution'] = [float(res['PixelSizeUm']),float(res['PixelSizeUm']),np.sin(head["angle"]*np.pi/180)*head["stageStepSize"]]
    head['type'] = res['PixelType']
    return head

args = sys.argv[1:]
start_time = time.time()
filelist = []
root = args[0] #directory containing raw tiff stacks
n_pos = int(args[1]) #number of position strips in root directory
n_seg = int(args[2]) #number of segments in each position to process
dataset = args[3] #dataset name for output folder

out = list(map(methodcaller("split", "_Pos"), os.listdir(root)))
prefix = '' #file prefix, e.g: "high_res_2um_step_size_700um_y_delta_MMStack"
output_root =  "/ACdata/processed/"+dataset+"/"+prefix+"/n5/Full/"
overlap = None #estimated overlap between Positions
md = dict() #metadata
state = {"layers": []} #contains layers fo neuroglancer

#iterate through each Position to convert to 2D Tiffs
for Position in range(0,n_pos):
    filelist = list()
    for f in out:
        if len(f) == 2:
            if prefix == '':
                prefix = f[0]
            if int(re.split("[_.]",f[1])[0]) == Position:
                filelist.append(root+"/"+prefix+"_Pos"+f[1])
    total_segs = len(filelist)
    if n_seg > total_segs:
        print("ERROR: number of segments invalid")
    
    tiffdir = "/ACdata/processed/"+dataset+"/"+"/Tiff/2D/Pos%d/Full/"%(Position)
    os.makedirs(tiffdir, exist_ok=True)
    
    #convert tiff stacks to 2D
    files = natsorted(filelist, alg=ns.IGNORECASE)
    files = files[:n_seg]
    print("Processing Files...")
    print(files)
    #input("Press Enter to continue...")
    stripfile(files, tiffdir)
    print("Finished 2D tiffs for Pos%d"%(Position))

    outputdir = "%sPos%d"%(output_root,Position)
    os.makedirs(outputdir, exist_ok=True)

    #convert to fullres n5
    curdir = os.getcwd()
    os.chdir('/allen/programs/celltypes/workgroups/em-connectomics/analysis_group/forSharmi/axonal/n5-spark/startup-scripts/')
    os.system('python spark-local/slice-tiff-to-n5.py -i %s -n %s -o pos%d -b 64,64,64'%(tiffdir,outputdir,Position))
    print("Finished n5 conversion for Pos%d"%(Position))

    #convert to multires n5
    os.system('python spark-local/n5-scale-pyramid.py -n %s -i pos%d -f 2,2,2 -o multirespos%d'%(outputdir,Position,Position))
    os.chdir(curdir)
    print("Finished multiscale conversion for Pos%d"%(Position))
    
    #get metadata
    if overlap == None:
        overlap = get_pixel_overlap(root, prefix)
        print("overlap: %d"%(overlap))
        md = get_metadata(files[0], Position, prefix, overlap) 
    #add attributes and s0 for multiscale viewing
    os.chdir(output_root)
    attributes = open("attributes.json", "w")
    attributes.write('{"pixelResolution" : {"unit":"um","dimensions":[%f,%f,%f]},"scales":[[1,1,1],[2,2,2],[4,4,4],[8,8,8],[16,16,16],[32,32,32],[64,64,64],[128,128,128]]}'%(md['pixelResolution'][0], md['pixelResolution'][1], md['pixelResolution'][2]))
    attributes.close()
    os.system('cp attributes.json %s/multirespos%d'%(outputdir, Position))
    os.chdir(output_root + 'Pos%d/multirespos%d'%(Position, Position))
    os.system('ln -s ../pos%d s0'%(Position))
    os.chdir(output_root + 'Pos%d/pos%d'%(Position, Position))
    os.system('ln -s ../pos%d pos%d'%(Position, Position))
    os.chdir(curdir)

    #create ng link
    layer_info = {"type": "image"}
    layer_info["shaderControls"] = { "normalized": { "range": [ 0, 5000 ] }}
    url = "n5://http://bigkahuna.corp.alleninstitute.org"
    url = url + output_root + 'Pos%d/multirespos%d'%(Position, Position)
    layer_info["source"] = {"url": url}
    layer_info["name"] = "Pos%d"%(Position)
    layer_info["source"]["transform"] = {
        "matrix" : [
            [1, 0, 0, 0],
            [0, 1, 0, overlap*Position],
            [0, 0, 1, 0]
        ],
        "outputDimensions" : {"x":[md['pixelResolution'][0],"um"], "y":[md['pixelResolution'][1],"um"], "z":[md['pixelResolution'][2],"um"]}
    }
    state["layers"].append(layer_info)

encoded_url = make_neuroglancer_url(state)
os.chdir("/ACdata/processed/"+dataset+"/")
ng = open("nglink.json", "w")
ng.write(encoded_url)
ng.close()
print("Done! Neuroglancer Link:")
print(encoded_url)
print("--- %s seconds ---" % (time.time() - start_time))
