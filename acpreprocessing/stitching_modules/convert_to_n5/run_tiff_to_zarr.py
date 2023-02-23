# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 19:42:51 2023

@author: kevint
"""

from tiff_to_ngff import tiffdir_to_ngff_group

for i in range(0,41):
    gstr = 'highres_Pos' + str(i)
    tiffdir = 'Y:/workflow_data/iSPIM2/MNx_S3AG_230222_highres/' + gstr
    outzarr = 'H:/MNx_S3AG_230222_highres_crop/' + gstr
    grpname = gstr
    tiffdir_to_ngff_group(tiffdir,output_n5=outzarr,group_names=[grpname],output='zarr',concurrency=20,max_mip=5,deskew_str='ispim2')