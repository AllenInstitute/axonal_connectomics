#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 12:45:25 2021

@author: shubha.bhaskaran
"""
import sys
from stitching_modules.nglink import write_nglink
from stitching_modules.nglink import create_state
from stitching_modules.multiscale_viewing import multiscale

root = "/ACdata/processed/testnglink/n5/"
pixelResolution = [0.26, 0.26, 1]
Position = 0
overlap = 509.53846153846166

state = {"layers": []}
ds_name = 'testnglink'

multiscale.add_multiscale_attributes(root, pixelResolution,Position)
layer0 = create_state.create_layer(root, Position, overlap, pixelResolution)
create_state.add_layer(state, layer0)
write_nglink.write_url(ds_name, state)

