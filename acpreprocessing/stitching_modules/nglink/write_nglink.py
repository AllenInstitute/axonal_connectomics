import os
import json
import urllib.parse
from acpreprocessing.utils import io, make_tinyurl
import requests


# Creates neuroglancer url from statejson
def make_neuroglancer_url(state,
                          base_url="http://bigkahuna.corp.alleninstitute.org/"
                          "neuroglancer"):
    state_json = json.dumps(state, separators=(',', ':')).replace("'",'"')
    encoded_state_json = urllib.parse.quote(state_json, safe='/:"')
    new_url = f"{base_url}/#!{encoded_state_json}"
    return new_url


# Writes url to a text file in output directory
def write_url(outputDir, state, fname):
    encoded_url = make_neuroglancer_url(state)
    ff = os.path.join(outputDir, fname)
    io.save_file(ff, encoded_url)
    print("Done! Neuroglancer Link:")
    print(encoded_url)


# Converts ng link to tinyurl and then writes
# to a text file in output directory
def write_tinyurl(outputDir, state, fname):
    encoded_url = make_neuroglancer_url(state)
    url = make_tinyurl.make_tiny(encoded_url)
    ff = os.path.join(outputDir, fname)
    io.save_file(ff, url)
    print("Done! Neuroglancer Link:")
    print(url)


def create_viz_link_from_json(
    ngl_json,
    outputDir,
    fname,
    url="https://json.neurodata.io/v1",
    neuroglancer_link="http://bigkahuna.corp.alleninstitute.org/neuroglancer/#!"
):
    r = requests.post(url, json=ngl_json)
    json_url = r.json()["uri"]
    viz_link = f"{neuroglancer_link}{json_url}"
    ff = os.path.join(outputDir, fname)
    io.save_file(ff, viz_link)
    print("Done! New Neuroglancer Link:")
    print(viz_link)