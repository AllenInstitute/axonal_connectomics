import itertools

import pytest

import acpreprocessing.utils.convert
import acpreprocessing.downsampling

from .voltest import fullsize_img_volumes, generate_array


def get_ds_shape(arr, ds_factors=(1, 4, 4)):
    return tuple(map(
        lambda ix: ix[1]//(ds_factors[ix[0]]),
        enumerate(arr.shape)))


@pytest.mark.parametrize(
      "img_vol_fixture,stack_downsample_method",
      itertools.product(
          fullsize_img_volumes, [
              "legacy",
              "area_average_downsample",
              "block_reduce", None]))
def test_downsampling(img_vol_fixture, stack_downsample_method,
                      request, ds_factor=4):
    img_vol = request.getfixturevalue(img_vol_fixture)
    target_shape = get_ds_shape(img_vol, (1, ds_factor, ds_factor))

    ds_vol = acpreprocessing.utils.convert.downsample_stack(
        img_vol, ds_factor, method=stack_downsample_method)

    assert ds_vol.shape == target_shape

    # mean of even subsamples is mean of full sample
    assert ds_vol.mean() == img_vol.mean()


@pytest.mark.parametrize(
    "expected_func,ds_factor,vol_shape",
    [("acpreprocessing.utils.convert.area_average_downsample", 4, (5, 512, 512)),
     ("skimage.measure.block_reduce", 4, (5, 513, 513))])
def test_downsample_default(expected_func, ds_factor, vol_shape, mocker):
    vol = generate_array(vol_shape)
    mocked_f = mocker.patch(expected_func)
    mocked_f.assert_not_called()

    ds_vol = acpreprocessing.utils.convert.downsample_stack(
        vol, ds_factor, method=None)

    mocked_f.assert_called_once()
