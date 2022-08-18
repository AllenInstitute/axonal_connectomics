"""
Basic script to convert existing n5 data to 8bit
Used this for evaluation/experimentation of methods

@author: shbhas
"""

import numpy as np
import z5py

output_n5 = "/ACdata/processed/MN7_RH_3_9_S33_220405_high_res/MN7_RH_3_9_S33_220405_high_res.n5/"
out_8bit = "/ACdata/processed/MN7_RH_3_9_S33_220405_high_res8bit/MN7_RH_3_9_S33_220405_high_res8bit.n5"
fout = z5py.File(out_8bit)

for s in range(18,19):
    with z5py.File(output_n5, mode='r') as f:
        s0shape = f[f"setup{s}"]['timepoint0']["s5"].shape
        t = 4
        u = 9
        xyf = s0shape[1]/t
        zf = s0shape[0]/u
        print(s0shape)
        num_ints = np.iinfo(np.uint16).max + 1
        lut = np.uint8(np.sqrt(np.arange(num_ints)))

        fout.create_group(f"setup{s}")
        fout[f"setup{s}"].create_group("timepoint0")
        ds = fout[f"setup{s}"]["timepoint0"].create_dataset("s5", shape=s0shape, chunks = (int(zf),int(xyf),int(xyf)), dtype='uint8')
        
        for x in range(t):
            for y in range(t):
                for z in range(u):
                    a = int(z*zf)
                    bx = int(x*xyf)
                    by = int(y*xyf)
                    imvol = f[f"setup{s}"]['timepoint0']["s5"][a:int(a+zf), bx:int(bx+xyf), by:int(by+xyf)]
                    fout[f"setup{s}"]["timepoint0"]["s5"][a:int(a+zf), bx:int(bx+xyf), by:int(by+xyf)] = lut[imvol]
                    # print((a,int(a+zf),bx,int(bx+xyf),by,int(by+xyf)))