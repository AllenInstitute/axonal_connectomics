# code based on Russel's em-stitch notebook for lens correction
# defines a sift_detector class and utility functions

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def generate_rois_from_pointmatches(pm_list,axis_range,roi_dims,**kwargs):
    roilist = []
    for pm in pm_list:
        p_siftpts = pm["p_pts"]
        z,roipts = get_roipoints_from_siftpoints(p_siftpts,axis_range,roi_dims[0])
        if len(roipts)==0:
            print("no roi points found")
            roilist.append(None)
        else:
            if len(roipts.shape) == 1:
                roipts = roipts[np.newaxis,:]
            ptsmean = np.mean(roipts,axis=0)
            if roi_dims[1] is None:
                x = int(ptsmean[2])
                roi = [[z,z+roi_dims[0]],[],[x,x+roi_dims[2]]]
            elif roi_dims[2] is None:
                y = int(ptsmean[1])
                roi = [[z,z+roi_dims[0]],[y,y+roi_dims[1]],[]]
            roilist.append(roi)
    return roilist


def get_roipoints_from_siftpoints(p_siftpts,axis_range,roi_length):
    if axis_range is None:
        axis_range = [[] for i in range(p_siftpts.shape[1])]
    if len(axis_range[0]) == 0:
        axis_range[0] = [int(np.min(p_siftpts[:,0])),int(np.max(p_siftpts[:,0]))]
    zstarts = np.arange(axis_range[0][0],axis_range[0][1],int(roi_length/2))
    p_pts = []
    z = zstarts[0]
    for zs in zstarts:
        r = np.full(p_siftpts.shape[0],True)
        for ai,a in enumerate(axis_range):
            if len(a)>0:
                r = r & ((p_siftpts[:,ai]>=a[0]) & (p_siftpts[:,ai]<=a[1]))
            elif ai == 0:
                r = r & ((p_siftpts[:,ai]>=zs) & (p_siftpts[:,ai]<=zs+roi_length))
        pr = p_siftpts[r]
        if len(pr) > len(p_pts):
            p_pts = pr
            z = zs
    return z,p_pts


def stitch_over_segments(sd_kwargs,p_dslist,q_dslist,zstarts,zlength,i_slice,ij_shift,ns,ds,s0=0,**kwargs):
    ''' stitch by segment for ispim data (legacy, axis_type = "ispim")
    '''
    xdim = p_dslist[0].shape[4]
    roi_list = [[[z,z+zlength],i_slice,[0,xdim]] for z in zstarts]
    p_ptlist,q_ptlist = stitch_over_rois(sd_kwargs=sd_kwargs,
                                         p_dslist=p_dslist,
                                         q_dslist=q_dslist,
                                         axis_type="ispim",
                                         roi_list=roi_list,
                                         ij_shift=ij_shift,
                                         ns=ns,
                                         ds=ds,
                                         s0=s0)
    return p_ptlist,q_ptlist


def stitch_over_rois(sd_kwargs,p_dslist,q_dslist,axis_type,roi_list,ij_shift,ns,ds,s0=0,**kwargs):
    '''
    Parameters
    ----------
    p_dslist : TYPE
        DESCRIPTION.
    q_dslist : TYPE
        DESCRIPTION.
    axis_type : TYPE
        DESCRIPTION.
    roi_list : TYPE
        DESCRIPTION.
    stitch_axes : TYPE
        DESCRIPTION.
    ij_shift : TYPE
        DESCRIPTION.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    p_ptlist,q_ptlist: lists of ndarray zyx coordinates of point matches for each pq pair in dslists

    '''
    pmlist = []
    sd = SiftDetector(**sd_kwargs)
    for p_src,q_src,roi in zip(p_dslist,q_dslist,roi_list):
        if not roi is None:
            # k1_tot,k2_tot,j_slice = self.zy_stitch(p_src,q_src,roi[0][0],roi[0][1],roi[2][0],roi[2][0]+ij_shift,**kwargs)
            if axis_type == "ispim":
                axis = 1
                z0,z1 = roi[0]
                py = roi[1]
                qs = roi[1] + ij_shift + s0
                x0,x1 = roi[2]
                p_img = p_src[0,0,z0:z1,py,x0:x1]
                q_stack = q_src[0,0,z0:z1,(qs-ns*ds):(qs+(ns+1)*ds):ds,x0:x1]
                k2_add = np.array([0,0])
            elif axis_type == "zyx":
                axis = 2
                z0,z1 = roi[0]
                y0,y1 = roi[1]
                px = roi[2]
                qs = roi[2] + s0
                p_img = p_src[0,0,z0:z1,y0:y1,px]
                q_stack = q_src[0,0,z0:z1,y0+ij_shift:y1+ij_shift,(qs-ns*ds):(qs+(ns+1)*ds):ds]
                k2_add = np.array([ij_shift,0])
            elif axis_type == "xzy":
                axis = 2
                z0,z1 = roi[1]
                y0,y1 = roi[2]
                px = roi[0]
                qs = roi[0] + s0
                p_img = p_src[0,0,px,z0:z1,y0:y1]
                q_stack = q_src[0,0,(qs-ns*ds):(qs+(ns+1)*ds):ds,z0:z1,y0+ij_shift:y1+ij_shift].transpose((1,2,0))
                k2_add = np.array([ij_shift,0])
            k1_tot,k2_tot,best_slice = sd.detect_in_best_slice(p_img, q_stack, axis=axis, **kwargs)
            k2_tot += k2_add
            if not k1_tot is None:
                k2_slice = qs - ns*ds + ds*best_slice
                pmlist.append((k1_tot,k2_tot,k2_slice))
            else:
                pmlist.append(None)
        else:
            pmlist.append(None)
    if pmlist:
        pq_lists = [[],[]]
        for s,roi in zip(pmlist,roi_list):
            for ii,t in enumerate(pq_lists):
                if not s is None:
                    k = np.empty((s[ii].shape[0],3))
                    if axis_type == "ispim":
                        zi = roi[0][0]
                        xi = roi[2][0]
                        y = roi[1]
                        k[:,0] = s[ii][:,1] + zi
                        k[:,1] = y if ii == 0 else s[2]
                        k[:,2] = s[ii][:,0] + xi
                    elif axis_type == "zyx":
                        zi = roi[0][0]
                        yi = roi[1][0]
                        x = roi[2]
                        k[:,0] = s[ii][:,1] + zi
                        k[:,1] = s[ii][:,0] + yi
                        k[:,2] = x if ii == 0 else s[2]
                    elif axis_type == "xzy":
                        x = roi[0]
                        zi = roi[1][0]
                        yi = roi[2][0]
                        k[:,0] = x if ii == 0 else s[2]
                        k[:,1] = s[ii][:,1] + zi
                        k[:,2] = s[ii][:,0] + yi
                    t.append(k)
                else:
                    t.append(None)
        p_ptlist = pq_lists[0]
        q_ptlist = pq_lists[1]
        return p_ptlist,q_ptlist
    else:
        return None,None


class SiftDetector(object):
    def __init__(self,clahe_kwargs,sift_kwargs,flann_args,ratio=0.7,min_inliers=100):
        # CLAHE equalizer
        self.clahe = cv.createCLAHE(**clahe_kwargs)
        # Initiate SIFT detector
        self.sift = cv.SIFT_create(**sift_kwargs)
        # FLANN feature matcher
        self.flann = cv.FlannBasedMatcher(*flann_args)
        self.ratio = ratio
        self.minin = min_inliers
        
    def detect_keypoints(self,img):
        cimg = self.clahe.apply(img)
        cimg = np.sqrt(cimg).astype('uint8')
        kp, des = self.sift.detectAndCompute(cimg,None)
        return kp,des,cimg
    
    
    def compute_matches(self,kp1,des1,kp2,des2,draw=False,cimg1=None,cimg2=None):
        matches = self.flann.knnMatch(des1,des2,k=2)
        # Need to draw only good matches, so create a mask
        matchesMask = [[0,0] for i in range(len(matches))]
        # ratio test as per Lowe's paper
        good = []
        for i,(m,n) in enumerate(matches):
            if m.distance < self.ratio*n.distance:
                matchesMask[i]=[1,0]
                good.append(m)
        #print(len(good))
        k1 = []
        k2 = []
        if len(good)>self.minin:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
            M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
            matchesMask = mask.ravel().tolist()
            if draw and not cimg1 is None and not cimg2 is None:
                self.draw_matches(cimg1,kp1,cimg2,kp2,good,matchesMask)
            goodMask = np.asarray(good)[np.asarray(matchesMask).astype('bool')]
            imgIdx = np.array([g.imgIdx for g in goodMask])
            tIdx = np.array([g.trainIdx for g in goodMask])
            qIdx = np.array([g.queryIdx for g in goodMask])
            k1xy = np.array([np.array(k.pt) for k in kp1])
            k2xy = np.array([np.array(k.pt) for k in kp2])

            for i in range(len(tIdx)):
                if imgIdx[i] == 1:
                    k1.append(k1xy[tIdx[i]])
                    k2.append(k2xy[qIdx[i]])
                else:
                    k1.append(k1xy[qIdx[i]])
                    k2.append(k2xy[tIdx[i]])
            if len(k1)>0:
                k1 = np.array(k1)
                k2 = np.array(k2)
            # limit number of matches to random subset if too many
                if k1.shape[0] > 10000:
                    a = np.arange(k1.shape[0])
                    np.random.shuffle(a)
                    k1 = k1[a[0: 10000], :]
                    k2 = k2[a[0: 10000], :]
        return k1,k2,good,matchesMask
        
        
    def draw_matches(self,cimg1,kp1,cimg2,kp2,good,mask=[],**draw_kwargs):
        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                        singlePointColor = None,
                        matchesMask = mask, # draw only inliers
                        flags = 2)
        draw_params.update(draw_kwargs)
        cvimg = cv.drawMatches(cimg1,kp1,cimg2,kp2,good,None,**draw_params)
        return cvimg
        
    
    def detect_and_combine(self,kp1,des1,cimg1,imgStack,draw=False,axis=2,max_only=False):
        k1_tot = []
        k2_tot = []
        good = np.zeros(imgStack.shape[axis],dtype=int)
        islice = []
        for i in range(imgStack.shape[axis]):
            if axis==2:
                img = imgStack[:,:,i]
            elif axis==1:
                img = imgStack[:,i,:]
            kp2,des2,cimg2 = self.detect_keypoints(img)
            k1,k2,_,__ = self.compute_matches(kp1,des1,kp2,des2)
            if isinstance(k1, np.ndarray):
                k1_tot.append(k1)
                k2_tot.append(k2)
                good[i] = k1.shape[0]
                islice.append(np.ones(k1.shape,dtype=int)*i)
        if k1_tot:
            if max_only:
                imax = np.argmax(good[good>0])
                return k1_tot[imax],k2_tot[imax],good,islice[imax]
            else:
                return np.concatenate(k1_tot),np.concatenate(k2_tot),good,np.concatenate(islice)
        else:
            return None,None,None,None
        
    
    def detect_in_best_slice(self,p_img,q_stack,axis,scatter=False,**kwargs):
        # ji = q_slice
        # imgRef = p_src[0,0,z0:z1,:,p_slice]
        # detect SIFT keypoints for reference slice
        kp1, des1, cimg1 = self.detect_keypoints(p_img)
        # imgStack = q_src[0,0,z0:z1,:,(ji-nx*dx):(ji+(nx+1)*dx):dx]
        # detect correspondences in slices from neighboring strip
        k1_tot,k2_tot,good,k2slice = self.detect_and_combine(kp1,des1,cimg1,q_stack,False,axis=axis,max_only=True)
        print("Number of correspondences: " + str(good))
        if not k1_tot is None and k1_tot.shape[0]>10:
            k = k2_tot-k1_tot
            print('total correspondences for analysis: ' + str(k.shape[0]))
            # estimate stitching translation with median displacement
            km = np.median(k,axis=0)
            print('median pixel displacements:' + str(km))
            # display scatter of displacements around median estimate
            if scatter:
                plt.scatter(k[:,0]-km[0],k[:,1]-km[1],s=1)
                plt.xlim((-5,5))
                plt.ylim((-5,5))
                plt.show()
            # identify slice index with most correspondences
            # j_slice = ji - nx*dx + dx*np.argmax(good)
            best_slice = np.argmax(good)
            return k1_tot,k2_tot,best_slice
        else:
            print("not enough correspondences")
            return None,None,None