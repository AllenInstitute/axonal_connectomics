"""
Basic script to split channels of acquisition (on tiff level)
Used this before channel splitting was incorporated into n5 conversion

@author: shbhas
"""

import os
from natsort import natsorted, ns
from acpreprocessing.utils import io
from acpreprocessing.utils.metadata import parse_metadata

def sort_files(filedir):
    filelist = os.listdir(filedir)
    return natsorted(filelist, alg=ns.IGNORECASE)

outputdir = "/ACdata/processed/iSPIM2/MN7_RH_3_9_S19_220606_high_res/split_channels/"
inputdir = "/ispim2_data/workflow_data/iSPIM2/MN7_RH_3_9_S19_220606_high_res/"
md_input = {
            "rootDir": "/ispim2_data/workflow_data/iSPIM2/MN7_RH_3_9_S19_220606_high_res/",
            "fname": "acqinfo_metadata.json"
            }

n_pos = parse_metadata.ParseMetadata(input_data=md_input).get_number_of_positions()
print(n_pos)
if not os.path.isdir(outputdir):
    os.makedirs(outputdir)

for s in range(n_pos):
    index = 0
    posdir = inputdir+f"high_res_Pos{s}/"
    for inputfile in sort_files(posdir):
        print(inputfile)
        if inputfile[0]=='.':
            continue
        I = io.get_tiff_image(posdir+inputfile)
        if index%2==0:
            chan0 = I[0::2, : , :]
            chan1 = I[1::2, : , :]
        else:
            chan1 = I[0::2, : , :]
            chan0 = I[1::2, : , :]
        fname0 = outputdir + f"ch0/high_res_Pos{s}/"+"{0:05d}.tif".format(index)
        fname1 = outputdir + f"ch1/high_res_Pos{s}/"+"{0:05d}.tif".format(index)
        print(fname0)
        io.save_tiff_image(chan0, fname0)
        io.save_tiff_image(chan1, fname1)
        index += 1