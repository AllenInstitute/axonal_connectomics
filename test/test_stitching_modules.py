import pytest

try:
    import acpreprocessing.stitching_modules.convert_to_n5.tiff_to_n5
except ImportError:
    # FIXME: z5py installs with conda, not currently configured on ci
    pass
from acpreprocessing.stitching_modules.metadata import parse_metadata
import acpreprocessing.stitching_modules.multiscale_viewing.multiscale
import acpreprocessing.stitching_modules.stitch.create_json
import acpreprocessing.stitching_modules.stitch.stitch
from acpreprocessing.stitching_modules.nglink import create_layer
from acpreprocessing.stitching_modules.nglink import update_state
import acpreprocessing.stitching_modules.nglink.create_nglink
import acpreprocessing.stitching_modules.nglink.write_nglink
from acpreprocessing.utils import io
import os


# skipped because we have changed the way layers are defined without updating this test
@pytest.mark.skip
# Test nglink.create_layer.create_layer
@pytest.mark.parametrize("outputDir, position, overlap, pixelResolution",
                         [("/testout", 2, 200, [0.1, 0.1, 0.1]),
                          ("/test2/testout/", 20, 500, [0.406, 0.406, 1.0])])
def test_create_layer(outputDir, position, overlap, pixelResolution):
    layer = create_layer.create_layer(outputDir, position,
                                      overlap, pixelResolution)
    url = "n5://http://bigkahuna.corp.alleninstitute.org"
    url = url + outputDir + '/Pos%d.n5/multirespos%d' % (position, position)
    assert layer["source"]["url"] == url
    assert layer["source"]["transform"]["matrix"][1][3] == overlap*position
    assert layer["source"]["transform"]["outputDimensions"]["x"][0] == pixelResolution[0]
    assert layer["source"]["transform"]["outputDimensions"]["y"][0] == pixelResolution[1]
    assert layer["source"]["transform"]["outputDimensions"]["z"][0] == pixelResolution[2]


# Test update state
@pytest.mark.parametrize("x, y, z, overlap, factor", [(30, 500, 2, 1000, 2)])
def test_update_state(x, y, z, overlap, factor):
    state = {'layers': []}
    state['layers'].append({'source': {'transform': {'matrix': [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0]
            ]}}})
    state['layers'][0]['source']['transform']['matrix'][0][3] = 0
    state['layers'][0]['source']['transform']['matrix'][1][3] = overlap
    state['layers'][0]['source']['transform']['matrix'][2][3] = 0
    stitchoutjson = [{"type": "GRAY16", "index": 0, "file": "test",
                      "position": [],
                      "size": [288, 288, 1960],
                      "pixelResolution":[0.406, 0.406, 2]}]
    stitchoutjson[0]['position'].append(x)
    stitchoutjson[0]['position'].append(y)
    stitchoutjson[0]['position'].append(z)
    update_state.update_positions(state, stitchoutjson, 1, factor)
    assert state['layers'][0]['source']['transform']['matrix'][0][3] == x*factor
    assert state['layers'][0]['source']['transform']['matrix'][1][3] == y*factor
    assert state['layers'][0]['source']['transform']['matrix'][2][3] == z*factor
