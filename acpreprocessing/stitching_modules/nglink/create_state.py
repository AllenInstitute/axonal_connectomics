
def create_layer(output_root, Position, overlap, pixelResolution):
    layer_info = {"type": "image"}
    layer_info["shaderControls"] = { "normalized": { "range": [ 0, 5000 ] }}
    url = "n5://http://bigkahuna.corp.alleninstitute.org"
    url = url + output_root + 'Pos%d/multirespos%d'%(Position, Position)
    layer_info["source"] = {"url": url}
    layer_info["name"] = "Pos%d"%(Position)
    layer_info["source"]["transform"] = {
        "matrix" : [
            [1, 0, 0, 0],
            [0, 1, 0, overlap*Position],
            [0, 0, 1, 0]
        ],
        "outputDimensions" : {"x":[pixelResolution[0],"um"], "y":[pixelResolution[1],"um"], "z":[pixelResolution[2],"um"]}
    }
    return layer_info

def add_layer(state, layer_info):
	state["layers"].append(layer_info)
