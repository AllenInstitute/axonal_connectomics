# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 13:33:46 2023

@author: kevint
"""

import json
import gzip
import numpy

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
    
def save_pointmatch_file(pmdata,jsonpath):
    with gzip.open(jsonpath, 'w') as fout:
        fout.write(json.dumps(pmdata,cls=NumpyEncoder).encode('utf-8'))
        

def read_pointmatch_file(jsonpath):
    with gzip.open(jsonpath, 'r') as fin:
        data = json.loads(fin.read().decode('utf-8'))
    return data