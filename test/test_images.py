import io
import json
import tempfile

import imageio
import numpy
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


@pytest.mark.parametrize("img_fixture", fullsize_imgs)
def test_io_rw(tmp_path, img_fixture, request):
    img = request.getfixturevalue(img_fixture)
    with tempfile.NamedTemporaryFile(
            dir=tmp_path, prefix=img_fixture,
            suffix=".tif") as tmpf:

        rimg = acpreprocessing.utils.io.get_tiff_image(img)

        acpreprocessing.utils.io.save_tiff_image(rimg, tmpf.name)

        rwimg = acpreprocessing.utils.io.get_tiff_image(tmpf.name)

        assert rwimg.dtype == rimg.dtype
        assert numpy.array_equal(rimg, rwimg)


@pytest.mark.parametrize("img_fixture", fullsize_imgs)
def test_get_metadata(tmp_path, img_fixture, request):
    img = request.getfixturevalue(img_fixture)

    md = acpreprocessing.utils.io.get_metadata(img)

    assert isinstance(md, dict)


@pytest.mark.parametrize("img_fixture", fullsize_imgs)
def test_write_md(tmp_path, img_fixture, request):
    img = request.getfixturevalue(img_fixture)
    with tempfile.NamedTemporaryFile(
            dir=tmp_path, prefix=img_fixture,
            suffix=".json") as tmpf:

        md = acpreprocessing.utils.io.get_metadata(img)

        acpreprocessing.utils.io.save_metadata(tmpf.name, md)

        with open(tmpf.name, "r") as f:
            md_written = json.load(f)
        assert md_written == md
