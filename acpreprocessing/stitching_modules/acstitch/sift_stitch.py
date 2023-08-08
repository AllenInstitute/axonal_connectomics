# code based on Russel's em-stitch notebook for lens correction
# defines a sift_detector class and utility functions

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

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
        cimg1 = self.clahe.apply(img)
        cimg1 = np.sqrt(cimg1).astype('uint8')
        kp1, des1 = self.sift.detectAndCompute(cimg1,None)
        return kp1,des1,cimg1
    
    def detect_matches(self,kp1,des1,cimg1,img,draw=False):
        cimg2 = self.clahe.apply(img)
        cimg2 = np.sqrt(cimg2).astype('uint8')
        # find the keypoints and descriptors with SIFT
        kp2, des2 = self.sift.detectAndCompute(cimg2,None)
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
        k1xy = np.array([np.array(k.pt) for k in kp1])
        k2xy = np.array([np.array(k.pt) for k in kp2])
        k1 = []
        k2 = []
        if len(good)>self.minin:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
            M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
            matchesMask = mask.ravel().tolist()
            if draw:
                draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                               singlePointColor = None,
                               matchesMask = matchesMask, # draw only inliers
                               flags = 2)
                img4 = cv.drawMatches(cimg1,kp1,cimg2,kp2,good,None,**draw_params)
                plt.figure(figsize=(20,20))
                plt.imshow(img4, 'gray'),plt.show()

            good = np.array(good)[np.array(matchesMask).astype('bool')]
            imgIdx = np.array([g.imgIdx for g in good])
            tIdx = np.array([g.trainIdx for g in good])
            qIdx = np.array([g.queryIdx for g in good])
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

        return k1,k2
    
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
            k1,k2 = self.detect_matches(kp1,des1,cimg1,img,draw)
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
        
        
    def stitch_over_segments(self,slice_axes,p_dslist,q_dslist,zstarts,zlength,**kwargs):
        pmlist = []
        if slice_axes == "zx":
            for zs in zstarts:
                seglist = self.run_zx_stitch(p_dslist,q_dslist,zs,zs+zlength,**kwargs)
                pmlist.append(seglist)
        if pmlist:
            n_tiles = len(pmlist[0][0])
            tile_pmlist = [[[] for i in range(n_tiles)],[[] for i in range(n_tiles)]]
            for pm in pmlist:
                for i in range(2):
                    for ii in range(n_tiles):
                        if not pm[i][ii] is None:
                            tile_pmlist[i][ii].append(pm[i][ii])
            p_ptlist = [np.concatenate(pm) if pm else [] for pm in tile_pmlist[0]]
            q_ptlist = [np.concatenate(pm) if pm else [] for pm in tile_pmlist[1]]
            return p_ptlist,q_ptlist
        else:
            return None,None

        
    def run_zx_stitch(self,
                      p_srclist,
                      q_srclist,
                      z0,z1,i_slice,j_slice,ny,dy,
                      scatter=False):
        # estimate translation between strips with the median 2D displacement of matching point correspondences
        siftklist = []
        j_slices = np.ones(len(p_srclist),dtype=int)*j_slice
        for i,dsRef in enumerate(p_srclist):
            ji = j_slices[i]
            # iterate over each strip and its subsequent neighbor to look for correspondences and estimate median
            dsStack = q_srclist[i]
            imgRef = dsRef[0,0,z0:z1,i_slice,:]
            # detect SIFT keypoints for reference slice
            kp1, des1, cimg1 = self.detect_keypoints(imgRef)
            imgStack = dsStack[0,0,z0:z1,(ji-ny*dy):(ji+(ny+1)*dy):dy,:]
            # detect correspondences in slices from neighboring strip
            k1_tot,k2_tot,good,k2slice = self.detect_and_combine(kp1,des1,cimg1,imgStack,False,axis=1,max_only=True)
            print("Number of correspondences: " + str(good))
            if not k1_tot is None and k1_tot.shape[0]>200:
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
                j_slices[i] = ji - ny*dy + dy*np.argmax(good)
                siftklist.append((k1_tot,k2_tot,k2slice))
            else:
                print("not enough correspondences for strip " + str(i))
                siftklist.append(None)
        siftstitch = [[],[]]
        for i,s in enumerate(siftklist):
            zi = z0
            yi = j_slices[i]
            for ii,t in enumerate(siftstitch):
                if not s is None:
                    kzyx = np.empty((s[ii].shape[0],3))
                    kzyx[:,0] = s[ii][:,0] + zi
                    kzyx[:,1] = i_slice if ii == 0 else yi
                    kzyx[:,2] = s[ii][:,1]
                    t.append(kzyx)
                else:
                    t.append(None)
        return siftstitch

    def run_zy_stitch(self,
                      p_srclist,
                      q_srclist,
                      z0,z1,i_slice,j_slice,nx,dx,
                      scatter=False):
        # estimate translation between strips with the median 2D displacement of matching point correspondences
        siftklist = []
        j_slices = np.ones(len(p_srclist),dtype=int)*j_slice
        for i,dsRef in enumerate(p_srclist):
            ji = j_slices[i]
            # iterate over each strip and its subsequent neighbor to look for correspondences and estimate median
            dsStack = q_srclist[i]
            imgRef = dsRef[0,0,z0:z1,:,i_slice]
            # detect SIFT keypoints for reference slice
            kp1, des1, cimg1 = self.detect_keypoints(imgRef)
            imgStack = dsStack[0,0,z0:z1,:,(ji-nx*dx):(ji+(nx+1)*dx):dx]
            # detect correspondences in slices from neighboring strip
            k1_tot,k2_tot,good,k2slice = self.detect_and_combine(kp1,des1,cimg1,imgStack,False,axis=2,max_only=True)
            print("Number of correspondences: " + str(good))
            if not k1_tot is None and k1_tot.shape[0]>100:
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
                j_slices[i] = ji - nx*dx + dx*np.argmax(good)
                siftklist.append((k1_tot,k2_tot,k2slice))
            else:
                print("not enough correspondences for strip " + str(i))
                siftklist.append(None)

        siftstitch = [[],[]]
        for i,s in enumerate(siftklist):
            zi = z0
            xi = j_slices[i]
            for ii,t in enumerate(siftstitch):
                if not s is None:
                    kzyx = np.empty((s[ii].shape[0],3))
                    kzyx[:,0] = s[ii][:,0] + zi
                    kzyx[:,1] = s[ii][:,1]
                    kzyx[:,2] = i_slice if ii == 0 else xi
                    t.append(kzyx)
                else:
                    t.append(None)
        return siftstitch