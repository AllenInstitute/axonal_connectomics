import numpy as np
from scipy import interpolate, signal
import os
import time

import imageio as iio
import z5py

example_input = {
    'input_dir': '/ACdata/processed/MN8_RH_S11_220214_high_res/stitch-s2/export.n5',
    'output_n5': '',
    'output_tif': '/ACdata/samk/flatten_tif_data/MN8_RH_S11_220214_high_res.tif',
    'flat_side': 'top',
    'global_thr': 32,
    'navg': 32,
    'nzout': 100,
    'ztol': 0.6,
    'npre': 0,
}

class FlattenVolumeModule:
    def __init__(self, input_dict=None):
        if input_dict is None:
            raise ValueError("ERROR: input_dict must be populated for now")
        
        self.input_dir = input_dict['input_dir']
        self.output_n5 = input_dict['output_n5']
        self.output_tif = input_dict['output_tif']
        self.flat_side = input_dict['flat_side']
        self.global_thr = input_dict['global_thr']
        self.navg = input_dict['navg']
        self.nzout = input_dict['nzout']
        self.ztol = input_dict['ztol']
        self.npre = input_dict['npre']
    
    def read_volume(self):
        print('Reading volume...')
        vol_type = self.input_dir.split('.')[-1]
        if vol_type == 'n5':
            imvol = self._read_n5()

        else:
            imvol = self._read_tiff()
        
        imvol = imvol[:, ::2, ::2]
        num_ints = np.iinfo(np.uint16).max + 1
        lut = np.uint8(np.sqrt(np.arange(num_ints)))
        imvol = lut[imvol]
        imvol = np.transpose(imvol)

        return imvol

    def _import_image_sequence(self, file_path, file_ext):
        tif_files = [s for s in os.listdir(file_path) if os.path.splitext(s)[1] == file_ext]    
        tif_files = np.sort(tif_files)
        fname = file_path + '/' + tif_files[0]
        imtmp = np.array(iio.imread(fname))
        dims = imtmp.shape
        nframes = len(tif_files)
        imvol = np.zeros((nframes, dims[0], dims[1]), dtype=imtmp.dtype)

        for i in range(nframes):
            fname = file_path + '/' + tif_files[i]
            imtmp = np.array(iio.imread(fname))
            imvol[i, :, :] = imtmp
    
    def _read_n5(self):
        print('Reading N5 volume...')
        with z5py.File(self.input_dir, mode='r') as f:
            imvol = np.asarray(f['c0']['s0'])
        
        return imvol

    def _read_tiff(self, isimagesequence=True):
        if isimagesequence:
            imvol = self.import_image_sequence(self.input_dir, '.tif')
        else:
            imvol = np.array(iio.mimread(self.input_dir, memtest="20000MB"))
        
        return imvol

    def _remove_borders(self, imvol, bordercolor=64, background=32):
        borderinds = np.where(imvol[0, :, :] == bordercolor)
        imvol[:, borderinds[0], borderinds[1]] = background

        return imvol

    def _fast_downsample(self, imvol, binsize, method='mean'):
        dtypein = imvol.dtype
        dims = imvol.shape

        if len(dims) == 3:
            nframes = dims[0]
            nstartind = 1
        elif len(dims) == 2:
            nframes = 1
            nstartind = 0
        else:
            raise ValueError("in fastdownsample ndims not equal 2 or 3")
        
        nx = (dims[nstartind] // binsize)*binsize
        ny = (dims[nstartind+1] // binsize)*binsize
        nxsml = nx//binsize
        nysml = ny//binsize

        if len(dims) == 3:
            stacksml=np.zeros([nframes,nxsml,nysml] ,dtype=dtypein)
        else:
            stacksml=np.zeros([nxsml,nysml],dtype=dtypein)
        
        if method == 'mean':
            stacksml[:] = (imvol[:,0:nx,0:ny]).reshape([nframes,nxsml,binsize,nysml,binsize]).mean(4).mean(2) [:]
        if method == 'median':
            stacksml[:] = np.median(np.median((imvol[:, 0:nx, 0:ny]).reshape([nframes, nxsml, binsize, nysml, binsize]), 4), 2)[:]
    
        return stacksml

    def median_filter_2d(self, imvol, numfilt=3):
        dims = np.shape(imvol)
        dt = imvol.dtype
        stackout = np.zeros(dims, dtype=dt)
        for i in range(dims[0]):
            stackout[i,:,:] = signal.medfilt2d(imvol[i,:,:])
        return(stackout)
    
    def flatten(self, imvol):
        if self.flat_side == 'top' or self.flat_side =='bottom':
            imvol, z0s, imvolsml = self._flatten_one(imvol, side=self.flat_side)
        elif self.flat_side == 'both':
            imvol, z0s, imvolsml = self._flatten_both(imvol)
        else:
            raise NotImplementedError
        
        return imvol, z0s, imvolsml
            
    def _flatten_one(self, imvol, side='top', nmedfilt=5, method='mean'):
        dims = imvol.shape

        if side == 'bottom':
            imvol = np.flip(imvol,0)
        
        imvolsml = self._fast_downsample(imvol, self.navg, method=method)
        
        if nmedfilt > 1:
            imvolsml = self.median_filter_2d(imvolsml,
                numfilt=nmedfilt)

        z0s = np.argmax(imvolsml>self.global_thr,axis = 0)
        z0s[np.where(z0s > dims[0] - self.nzout)] = dims[0] - self.nzout

        xs = np.arange(np.int16(self.navg/2),2+dims[1]-self.navg/2,self.navg)
        ys = np.arange(np.int16(self.navg/2),2+dims[2]-self.navg/2,self.navg)
        interpfun = interpolate.interp2d(ys, xs, z0s)

        Xs = np.arange(dims[1])
        Ys = np.arange(dims[2])
        Z0s = np.uint16(interpfun(Ys, Xs))
        Z0s = Z0s.astype(np.uint32)
        newvol = np.zeros([self.nzout, dims[1] * dims[2]], dtype='uint8')
        imsize = int(dims[1]*dims[2])
        Zcurrent = np.arange(imsize)  + (Z0s.reshape([imsize])-self.npre) * imsize

        imvol = imvol.reshape(np.prod(dims))

        for i in range(self.nzout):
            newvol[i,:] = imvol[Zcurrent]
            Zcurrent += imsize # increment the indices by one frame
        imvol = imvol.reshape(dims)
        newvol = newvol.reshape([self.nzout,dims[1],dims[2]])

        Z0s = Z0s.reshape([dims[1],dims[2]])
        if side == 'bottom':
            imvol = np.flip(imvol, 0)
            imvolsml = np.flip(imvolsml, 0)
            newvol = np.flip(newvol, 0)
            Z0s[:] = dims[0] - Z0s[:]
        
        return newvol, Z0s, imvolsml

    def _flatten_both(self, imvol):
        raise NotImplementedError

    def write_output(self, imvol, z0s=None, imvolsml=None):
        #if len(self.output_n5) > 0:
        #    self._write_n5(imvol)
        
        if len(self.output_tif) > 0:
            self._write_tif(imvol, z0s=z0s, imvolsml=imvolsml)

    def _write_n5(self, imvol):
        raise NotImplementedError

    def _write_tif(self, imvol, z0s=None, imvolsml=None):
        frames = []
        for i in range(imvol.shape[0]):
            frames.append(imvol[i,:,:])

        fn_split = self.output_tif.split('.')
        iio.mimwrite(self.output_tif, imvol)

        if z0s is not None:  
            z0s_fn = fn_split[0] + '_z0s.' + fn_split[1]
            print(z0s_fn)
            iio.imwrite(z0s_fn, z0s)
        
        if imvolsml is not None:
            sml_frames = []
            for i in range(imvolsml.shape[0]):
                sml_frames.append(imvolsml[i,:,:])
            
            sml_fn = fn_split[0] + '_sml.' + fn_split[1]
            print(sml_fn)
            iio.mimwrite(sml_fn, imvolsml)

    def run(self):
        imvol = self.read_volume()

        imvol_flat, z0s, imvolsml = self.flatten(imvol)

        self.write_output(imvol_flat, z0s=z0s, imvolsml=imvolsml)

if __name__ == '__main__':
    mod = FlattenVolumeModule(input_dict=example_input)
    mod.run()