import io

import imageio
import pytest

import acpreprocessing.utils.ac_imageio
import acpreprocessing.utils.io
import acpreprocessing.header_parse

from .voltest import fullsize_imgs


@pytest.mark.parametrize("img_fixture", fullsize_imgs)
def test_actiff_vol_rw(img_fixture, request):
    img = request.getfixturevalue(img_fixture)
    rvol = imageio.volread(img, format="actiff")

    assert not {"description", "MicroManagerMetadata"} - rvol.meta.keys()

    with io.BytesIO() as b_io:
        imageio.volwrite(b_io, rvol, format="actiff")
        b = b_io.getvalue()
    wvol = imageio.volread(b, format="actiff")
    assert wvol.meta == rvol.meta
