
import numpy
import scipy.ndimage

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


def ccorr_disp(img1, img2, autocorrelation_threshold=None, padarray=False, value_threshold=0):
    if padarray:
        d = numpy.ceil(numpy.array(img1.shape) / 2)
        pw = numpy.asarray([(di,di) for di in d],dtype=int)
        img1 = numpy.pad(img1,pw)
        img2 = numpy.pad(img2,pw)
    if value_threshold:
        img1[img1<value_threshold] = 0
        img2[img2<value_threshold] = 0
    if autocorrelation_threshold is not None:
        cc, ac1, ac2 = ccorr_and_autocorr_fftn(img1, img2)
        ac1max = ac1.max()
        ac2max = ac2.max()
        if (not numpy.isnan(ac1max) and ac1max > 0) and (not numpy.isnan(ac2max) and ac2max > 0):
            autocorrelation_ratio = cc.max() / (numpy.sqrt(ac1max*ac2max))
            print(autocorrelation_ratio)
            if autocorrelation_ratio < autocorrelation_threshold:
                # what to do here?
                return None
        else:
            return None
    else:
        cc = ccorr_fftn(img1, img2)
    max_loc = subpixel_maximum(cc)
    mid_point = numpy.array(img1.shape) // 2
    return max_loc - mid_point


def get_point_correspondence(src_pt, dst_pt, src_patch, dst_patch, autocorrelation_threshold=0.8,padarray=False,value_threshold=0):
    disp = ccorr_disp(src_patch, dst_patch, autocorrelation_threshold, padarray,value_threshold)
    if disp is not None:
        return src_pt, dst_pt - disp
    return None,None