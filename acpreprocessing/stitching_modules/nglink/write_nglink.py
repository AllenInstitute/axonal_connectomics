import os
import json
import urllib.parse
from acpreprocessing.utils import io

def make_neuroglancer_url(state, base_url="http://neuroglancer-demo.appspot.com/"):
    state_json = json.dumps(state, separators=(',', ':')).replace("'",'"')
    encoded_state_json = urllib.parse.quote(state_json, safe='/:"')
    new_url = f"{base_url}/#!{encoded_state_json}"
    return new_url

def write_url(output_root, state, fname):
    encoded_url = make_neuroglancer_url(state)
    os.chdir(output_root)
    io.save_file(fname, encoded_url)
    print("Done! Neuroglancer Link:")
    print(encoded_url)

