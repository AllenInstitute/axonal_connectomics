import argparse
import json
import urllib.parse
import re

def make_neuroglancer_url(state, base_url="http://neuroglancer-demo.appspot.com/"):
    state_json = json.dumps(state, separators=(',', ':')).replace("'",'"')
    encoded_state_json = urllib.parse.quote(state_json, safe='/:"')
    new_url = f"{base_url}/#!{encoded_state_json}"
    return new_url


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('jsonfile', help="JSON file to read")
    args = parser.parse_args()

    state = {"layers": []}

    with open(args.jsonfile) as f:
        data = json.load(f)

    for info in data:
        layer_info = {"type": "image"}
        layer_info["shaderControls"] = { "normalized": { "range": [ 0, 5000 ] }}

        # /ispim1_data/processed/shubha/sharmidata/first/Pos0/multirespos0/s2
        # ... to ..
        # n5://http://bigkahuna.corp.alleninstitute.org/ispim1_data/processed/shubha/sharmidata/full/Pos0/multirespos0/

        url = "n5://http://bigkahuna.corp.alleninstitute.org"
        url = url + info["file"].strip("s2").replace("first","full")

        layer_info["source"] = {"url": url}

        # Try to find a name... Look for PosN in the file name
        layer_info["name"] = re.search("Pos(\d+)", info["file"]).group(0)

        # The transform will need to be fixed when the raw data has correct resolution metadata
        layer_info["source"]["transform"] = {
            "matrix" : [
                [1, 0, 0, info["position"][0]*2304/576*406],
                [0, 1, 0, info["position"][1]*2304/576*406],
                [0, 0, 1, info["position"][2]*406]
            ],
            "outputDimensions" : {"x":[1,"nm"], "y":[1,"nm"], "z":[1,"nm"]}
        }


        state["layers"].append(layer_info)

    encoded_url = make_neuroglancer_url(state)
    print(encoded_url)

    # TODO: Save JSON to a state server?

if __name__ == "__main__":
    main()

