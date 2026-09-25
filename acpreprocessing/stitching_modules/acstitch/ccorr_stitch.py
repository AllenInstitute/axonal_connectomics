import numpy
import scipy.ndimage
import cv2
    
    
    
def _safe_patch(ds, center, half_width, array_shape):
    """Extract a patch of size 2*half_width centered on `center`, zero-padding
    if the window would run outside array_shape."""
    lo = center - half_width
    hi = center + half_width
    # clip to valid array bounds
    lo_clipped = numpy.maximum(lo, 0)
    hi_clipped = numpy.minimum(hi, array_shape)
    sl = tuple(slice(int(l), int(h)) for l, h in zip(lo_clipped, hi_clipped))
    sub = ds[(0, 0) + sl]
    # figure out how much padding is needed on each side
    pad_before = lo_clipped - lo
    pad_after = hi - hi_clipped
    pad_width = [(int(b), int(a)) for b, a in zip(pad_before, pad_after)]
    if any(b or a for b, a in pad_width):
        sub = numpy.pad(sub, pad_width)
    return sub
    
    
    
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
    
    
def ccorr_disp(img1, img2, padarray=False):
    if padarray:
        d = numpy.ceil(numpy.array(img1.shape) / 2)
        pw = numpy.asarray([(di, di) for di in d], dtype=int)
        img1 = numpy.pad(img1, pw)
        img2 = numpy.pad(img2, pw)

    #p1 = numpy.percentile(img1, 70)
    #p2 = numpy.percentile(img2, 70)
    #img1[img1 < p1] = 0
    #img2[img2 < p2] = 0

    cc, ac1, ac2 = ccorr_and_autocorr_fftn(img1, img2)
    ac1max = ac1.max()
    ac2max = ac2.max()

    if (numpy.isnan(ac1max) or ac1max <= 0) or (numpy.isnan(ac2max) or ac2max <= 0):
        return None, numpy.nan

    ratio = cc.max() / numpy.sqrt(ac1max * ac2max)

    max_loc = subpixel_maximum(cc)
    mid_point = numpy.array(img1.shape) // 2
    disp = max_loc - mid_point

    return disp, ratio


def get_correspondences(A1_ds, A2_ds, A1_pts, A2_pts, w, r=1, pad=False, min_value=0, mad_thresh=3.0):
    w = numpy.asarray(w, dtype=int)
    if len(A1_pts.shape) < 2:
        A1_pts = numpy.array([A1_pts])
        A2_pts = numpy.array([A2_pts])
    A1_shape = numpy.asarray(A1_ds.shape[2:])
    A2_shape = numpy.asarray(A2_ds.shape[2:])

    print(A1_shape)

    # --- pass 1: compute displacement + ratio for every point, reject nothing yet ---
    p1_list = []
    p2_list = []
    disp_list = []
    ratio_list = []

    for p, q in zip(A1_pts.astype(int), A2_pts.astype(int)):
        print(p, q)
        A2sub = _safe_patch(A2_ds, q, r * w, A2_shape)
        A1sub = _safe_patch(A1_ds, p, w, A1_shape)
        if r > 1:
            pw = numpy.asarray([((r - 1) * wi, (r - 1) * wi) for wi in w], dtype=int)
            A1sub = numpy.pad(A1sub, pw)

        disp, ratio = ccorr_disp(A1sub, A2sub, padarray=pad)

        p1_list.append(p)
        p2_list.append(q)
        disp_list.append(disp)
        ratio_list.append(ratio)

    if not p1_list:
        return None, None

    ratios = numpy.array(ratio_list, dtype=float)

    # --- pass 2: reject outliers relative to the population's typical ratio ---
    valid = ~numpy.isnan(ratios) & numpy.array([d is not None for d in disp_list])
    if not numpy.any(valid):
        return None, None

    med = numpy.median(ratios[valid])
    mad = numpy.median(numpy.abs(ratios[valid] - med))
    scaled_mad = mad * 1.4826

    pm1 = []
    pm2 = []
    for p, q, disp, ratio in zip(p1_list, p2_list, disp_list, ratio_list):
        if disp is None or numpy.isnan(ratio):
            continue

        deviation = abs(ratio - med) / scaled_mad if scaled_mad > 0 else 0
        if deviation > mad_thresh:
            print(f"outlier: pt={p}, ratio={ratio:.3f}, median={med:.3f}, "
                  f"deviation={deviation:.2f} MADs")
            continue

        pm1.append(p)
        pm2.append(q - disp)

    if pm1:
        return numpy.asarray(pm1), numpy.asarray(pm2)
    return None, None