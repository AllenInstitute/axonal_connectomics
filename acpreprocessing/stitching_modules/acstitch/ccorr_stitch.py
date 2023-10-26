import numpy
from acpreprocessing.stitching_modules.acstitch.rtccorr import get_point_correspondence


def get_correspondences(A1_ds,A2_ds,A1_pts,A2_pts,w,r=1,pad=False,cc_threshold=0.8,min_value=0):
    cc_threshold /= r # TODO: is this the correct rescaling accounting for expanding reference?
    w = numpy.asarray(w,dtype=int)
    if len(A1_pts.shape)<2:
        A1_pts = numpy.array([A1_pts])
        A2_pts = numpy.array([A2_pts])
    pm1 = []
    pm2 = []
    for p,q in zip(A1_pts.astype(int),A2_pts.astype(int)):
        A2sub = A2_ds[0,0,(q-r*w)[0]:(q+r*w)[0],(q-r*w)[1]:(q+r*w)[1],(q-r*w)[2]:(q+r*w)[2]]
        A1sub = A1_ds[0,0,(p-w)[0]:(p+w)[0],(p-w)[1]:(p+w)[1],(p-w)[2]:(p+w)[2]]
        if r > 1:
            pw = numpy.asarray([((r-1)*wi,(r-1)*wi) for wi in w],dtype=int)
            A1sub = numpy.pad(A1sub,pw)
        p1,p2 = get_point_correspondence(p,q,A1sub,A2sub,autocorrelation_threshold=cc_threshold,padarray=pad,value_threshold=min_value)
        if not p1 is None:
            pm1.append(p1)
            pm2.append(p2)
    if pm1:
        return numpy.asarray(pm1),numpy.asarray(pm2)
    return None,None