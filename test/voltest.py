import collections
import contextlib

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


fullsize_img_volumes = [
    "uint16_image_volume_30_2048_2048"]

__all__ = fullsize_img_volumes
