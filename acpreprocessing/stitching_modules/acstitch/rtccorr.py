import numpy
import scipy.ndimage
import cv2

def correlate_fftns(fft1, fft2):
    prod = fft1 * fft2.conj()
    res = numpy.fft.ifftn(prod)
    
    corr = numpy.fft.fftshift(res).real
    return corr


def ccorr_fftn(img1, img2):
    # TODO do we want to pad this?
    fft1 = numpy.fft.fftn(img1)
    fft2 = numpy.fft.fftn(img2)
    
    return correlate_fftns(fft1, fft2)


def autocorr_fftn(img):
    fft = numpy.fft.fftn(img)
    return correlate_fftns(fft, fft)


def ccorr_and_autocorr_fftn(img1, img2):
    # TODO do we want to pad this?
    fft1 = numpy.fft.fftn(img1)
    fft2 = numpy.fft.fftn(img2)
    ccorr = correlate_fftns(fft1, fft2)
    acorr1 = correlate_fftns(fft1, fft1)
    acorr2 = correlate_fftns(fft2, fft2)
    return ccorr, acorr1, acorr2


def subpixel_maximum(arr):
    max_loc = numpy.unravel_index(numpy.argmax(arr), arr.shape)
    
    sub_arr = arr[
        tuple(slice(ml-1, ml+2) for ml in max_loc)
    ]
    
    # get center of mass of sub_arr
    subpixel_max_loc = numpy.array(scipy.ndimage.center_of_mass(sub_arr)) - 1
    return subpixel_max_loc + max_loc


