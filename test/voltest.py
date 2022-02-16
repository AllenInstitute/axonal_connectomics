import collections
import contextlib
import io
import json
import tempfile

import imageio
import numpy
import pytest

ImageTuple = collections.namedtuple("ImageTuple", ["filename", "arr", "repr"])


def generate_array(shape, minval=0, maxval=255, dtype='uint8'):
    return numpy.random.randint(minval, maxval, shape).astype(dtype)


@contextlib.contextmanager
def yield_array(*args, **kwargs):
    yield generate_array(*args, **kwargs)


@pytest.fixture(scope="function")
def uint16_image_volume_30_2048_2048():
    vol_shape = (30, 2048, 2048)
    with yield_array(
            vol_shape, minval=0, maxval=65535,
            dtype="uint16") as arr:
        yield arr


@contextlib.contextmanager
def yield_tiffimg_bytes_with_acqmd(vol_array, acquisition_tag=51123):
    acquisition_md = {"k": "v"}
    with io.BytesIO() as b_io:
        img_arr = imageio.core.util.Array(
            vol_array,
            {
                "description": "description",
                "extratags": [
                    (acquisition_tag, "s", 0, json.dumps(acquisition_md), True)
                ]
            })
        imageio.volwrite(b_io, img_arr, format="tiff")
        yield b_io.getvalue()


@contextlib.contextmanager
def yield_tiffimg_fn_with_acqmd(tmp_path, vol_array, acquisition_tag=51123):
    with yield_tiffimg_bytes_with_acqmd(vol_array, acquisition_tag) as b:
        with tempfile.NamedTemporaryFile(
                mode="wb", dir=tmp_path, suffix=".tif") as f:
            f.write(b)
            yield f.name


@pytest.fixture(scope="function")
def tiff_img_bytes_uint16_30_2048_2048_mdtag_51123(
        uint16_image_volume_30_2048_2048):
    with yield_tiffimg_bytes_with_acqmd(
            uint16_image_volume_30_2048_2048,
            acquisition_tag=51123) as b:
        yield b


@pytest.fixture(scope="function")
def tiff_img_file_uint16_30_2048_2048_mdtag_51123(
        tmp_path, uint16_image_volume_30_2048_2048):
    with yield_tiffimg_fn_with_acqmd(
            tmp_path, uint16_image_volume_30_2048_2048,
            acquisition_tag=51123) as fn:
        yield fn


@pytest.fixture(scope="function")
def tiff_img_bytes_uint16_30_2048_2048_mdtag_MicroManagerMetadata(
        uint16_image_volume_30_2048_2048):
    with yield_tiffimg_bytes_with_acqmd(
            uint16_image_volume_30_2048_2048,
            acquisition_tag="MicroManagerMetadata") as b:
        yield b


@pytest.fixture(scope="function")
def tiff_img_file_uint16_30_2048_2048_mdtag_MicroManagerMetadata(
        tmp_path, uint16_image_volume_30_2048_2048):
    with yield_tiffimg_fn_with_acqmd(
            tmp_path, uint16_image_volume_30_2048_2048,
            acquisition_tag="MicroManagerMetadata") as fn:
        yield fn


fullsize_img_volumes = [
    "uint16_image_volume_30_2048_2048"]

fullsize_imgs = [
    "tiff_img_bytes_uint16_30_2048_2048_mdtag_51123",
    "tiff_img_file_uint16_30_2048_2048_mdtag_51123",
    "tiff_img_bytes_uint16_30_2048_2048_mdtag_MicroManagerMetadata",
    "tiff_img_file_uint16_30_2048_2048_mdtag_MicroManagerMetadata",
    ]

__all__ = fullsize_imgs + fullsize_img_volumes
