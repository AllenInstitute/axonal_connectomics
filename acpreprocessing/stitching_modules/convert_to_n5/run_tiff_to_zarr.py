# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 19:42:51 2023

@author: kevint
"""

from tiff_to_zarr import tiffdir_to_n5_group

for i in range(0,39):
    gstr = 'highres_Pos' + str(i)
    tiffdir = 'Y:/workflow_data/iSPIM2/MNx_S3bB_230215_highres/' + gstr
    outzarr = 'H:/highres_crop/' + gstr
    grpname = gstr
    tiffdir_to_n5_group(tiffdir,output_n5=outzarr,group_names=[grpname],concurrency=20,max_mip=5,deskew_str='ispim2')