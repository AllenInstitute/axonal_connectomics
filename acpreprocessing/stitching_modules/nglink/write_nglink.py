import os
import json
import urllib.parse
from acpreprocessing.utils import io, make_tinyurl


def make_neuroglancer_url(state,
                          base_url="http://bigkahuna.corp.alleninstitute.org/"
                          "neuroglancer"):
    state_json = json.dumps(state, separators=(',', ':')).replace("'",'"')
    encoded_state_json = urllib.parse.quote(state_json, safe='/:"')
    new_url = f"{base_url}/#!{encoded_state_json}"
    return new_url


def write_url(output_root, state, fname):
    encoded_url = make_neuroglancer_url(state)
    ff = os.path.join(output_root, fname)
    io.save_file(ff, encoded_url)
    print("Done! Neuroglancer Link:")
    print(encoded_url)


def write_tinyurl(output_root, state, fname):
    encoded_url = make_neuroglancer_url(state)
    url = make_tinyurl.make_tiny(encoded_url)
    ff = os.path.join(output_root, fname)
    io.save_file(ff, url)
    print("Done! Neuroglancer Link:")
    print(url)
