import os
import json
import urllib.parse

def make_neuroglancer_url(state, base_url="http://neuroglancer-demo.appspot.com/"):
    state_json = json.dumps(state, separators=(',', ':')).replace("'",'"')
    encoded_state_json = urllib.parse.quote(state_json, safe='/:"')
    new_url = f"{base_url}/#!{encoded_state_json}"
    return new_url

def write_url(output_root, state):
	encoded_url = make_neuroglancer_url(state)
	os.chdir(output_root)
	ng = open("nglink.json", "w")
	ng.write(encoded_url)
	ng.close()
	print("Done! Neuroglancer Link:")
	print(encoded_url)


