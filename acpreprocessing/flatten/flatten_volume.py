import copy
import json
import numpy as np
from scipy import interpolate, signal
from scipy.ndimage import gaussian_filter, median_filter
import scipy.ndimage as ndimage
import os
import time

import imageio as iio
import z5py

import acpreprocessing
from acpreprocessing.stitching_modules.nglink import create_layer, create_nglink, update_state
import argschema

example_input = {
    "acq_id": "MN7_RH_3_2_S35_220127_high_res",
    "raw_tif_dir": "/ACdata/iSPIM2_ACDATA_temp",
    "input_dir": "/ACdata/processed/cutout_test/ELAST_0",
    "input_format": "n5",
    "output_dir": "/ACdata/samk/flatten/MN7/MN7_RH_3_2/cutout",
    "write_tif": True,
    "write_n5": False,
    "flat_side": "top",
    "global_thr": 28,
    "navg": 28,
    "nzout": 1000,
    "ztol": 0.6,
    "npre": 0
}

class FlattenVolumeModule:
    def __init__(self, input_dict=None):
        if input_dict is None:
            raise ValueError("ERROR: input_dict must be populated for now")
        
        self.input_dict = copy.deepcopy(input_dict)
        self.acq_id = input_dict['acq_id']
        self.raw_tif_dir = input_dict['raw_tif_dir']
        self.input_dir = input_dict['input_dir']
        self.input_format = input_dict['input_format']
        self.output_dir = input_dict['output_dir']
        self.write_tif = input_dict['write_tif']
        self.write_n5 = input_dict['write_n5']
        self.flat_side = input_dict['flat_side']
        self.global_thr = input_dict['global_thr']
        self.navg = input_dict['navg']
        self.nzout = input_dict['nzout']
        self.ztol = input_dict['ztol']
        self.npre = input_dict['npre']
    
    def read_volume(self):
        print('Reading volume...')
        if self.input_format == 'n5':
            imvol = self._read_n5()
        else:
            imvol = self._read_tiff()
        
        imvol = imvol[:, ::2, ::2]
        num_ints = np.iinfo(np.uint16).max + 1
        lut = np.uint8(np.sqrt(np.arange(num_ints)))
        imvol = lut[imvol]

        if self.input_format == 'n5':
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
        if os.path.exists(os.path.join(self.input_dir, self.acq_id, 'stitch-s2/export.n5')):
            print('stitch-s2 exists')
            with z5py.File(os.path.join(self.input_dir, self.acq_id, 'stitch-s2/export.n5'), mode='r') as f:
                imvol = np.asarray(f['c0']['s0'])
        else:
            print('stitch-s2 does not exist')
            with z5py.File(os.path.join(self.input_dir, self.acq_id, 'export.n5'), mode='r') as f:
                imvol = np.asarray(f['c0']['s0'])
        
        return imvol

    def _read_tiff(self, isimagesequence=False):
        print('Reading TIFF stack...')
        if isimagesequence:
            imvol = self._import_image_sequence(self.input_dir, '.tif')
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
        
        nx = (dims[nstartind] // binsize) * binsize
        ny = (dims[nstartind+1] // binsize) * binsize
        nxsml = nx // binsize
        nysml = ny // binsize

        if len(dims) == 3:
            stacksml=np.zeros([nframes,nxsml,nysml] ,dtype=dtypein)
        else:
            stacksml=np.zeros([nxsml,nysml],dtype=dtypein)
        
        if method == 'mean':
            stacksml[:] = (imvol[:,0:nx,0:ny]).reshape([nframes,nxsml,binsize,nysml,binsize]).mean(4).mean(2) [:]
        if method == 'median':
            stacksml[:] = np.median(np.median((imvol[:, 0:nx, 0:ny]).reshape([nframes, nxsml, binsize, nysml, binsize]), 4), 2)[:]
    
        return stacksml

    def median_filter_2d(self, imvol, zero_padding=100, numfilt=3):
        dims = np.shape(imvol)
        dt = imvol.dtype

        pad_dims = tuple(np.array(dims) + zero_padding*2)
        pad_vol = np.zeros(pad_dims)
        pad_vol[zero_padding:-zero_padding, zero_padding:-zero_padding, zero_padding:-zero_padding] = imvol[:,:,:]

        vol_out = np.zeros(pad_dims, dtype=dt)
        for i in range(dims[0]):
            vol_out[i,:,:] = signal.medfilt2d(pad_vol[i,:,:])
        
        return vol_out[zero_padding:-zero_padding, zero_padding:-zero_padding, zero_padding:-zero_padding]
    
    def ndfilter(self, img, sig=3):
        #img = ndimage.gaussian_filter(img, sigma=(sig, sig, 5), order=0)
        img = ndimage.median_filter(img, size=(sig, sig, 5))
        return img
    
    def flatten(self, imvol):
        if self.flat_side == 'top' or self.flat_side =='bottom':
            imvol, z0s, imvolsml = self._flatten_one(imvol, side=self.flat_side)
        elif self.flat_side == 'both':
            imvol, z0s, imvolsml = self._flatten_both(imvol)
        else:
            raise NotImplementedError
        
        return imvol, z0s, imvolsml
            
    def _flatten_one(self, imvol, side='top', nmedfilt=5, method='mean'):
        print('Flattening one side...')
        dims = imvol.shape

        if side == 'bottom':
            imvol = np.flip(imvol,0)
        
        imvolsml = self._fast_downsample(imvol, self.navg, method=method)
        #imvolsml = self.ndfilter(imvolsml)

        z0s = np.argmax(imvolsml > self.global_thr, axis=0)
        z0s[np.where(z0s > dims[0] - self.nzout)] = dims[0] - self.nzout

        xs = np.arange(np.int16(self.navg/2),2+dims[1]-self.navg/2,self.navg)
        ys = np.arange(np.int16(self.navg/2),2+dims[2]-self.navg/2,self.navg)
        print(xs.shape, ys.shape, z0s.shape)
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

    def write_output(self, imvol, z0s=None, imvolsml=None, write_json=True):
        print('Writing output')
        if self.write_n5:
            self._write_n5(imvol)
        
        if self.write_tif:
            self._write_tif(imvol, z0s=z0s, imvolsml=imvolsml)
        
        if write_json:
            with open(os.path.join(self.output_dir, self.acq_id) + '.json', 'w') as fp:
                json.dump(self.input_dict, fp, indent=4)

    def _write_n5(self, imvol):
        imvol = np.transpose(imvol)
        with z5py.File(os.path.join(self.output_dir, self.acq_id, 'flatten.n5'), mode='w') as f:
            imvol_n5 = f.create_dataset('data', shape=imvol.shape, dtype='uint16')
            imvol_n5[:] = imvol
        
        state = {"showDefaultAnnotations": False, "layers": []}
        
        layer_input = {
            "position": 0,
            "outputDir": self.output_n5,
            "rootDir": self.raw_tif_dir,
            }
        create_layer.NgLayer(input_data=layer_input).run(state)
        
        nglink_input = {
            "outputDir": self.output_n5,
            "fname": "nglink.txt"
        }

        if not os.path.exists(os.path.join(nglink_input['outputDir'], "nglink.txt")):
            create_nglink.Nglink(input_data=nglink_input).run(state)
        else:
            print("nglink.txt already exists!")

    def _write_tif(self, imvol, z0s=None, imvolsml=None):
        frames = []
        for i in range(imvol.shape[0]):
            frames.append(imvol[i,:,:])

        output_fn = os.path.join(self.output_dir, self.acq_id)
        print(output_fn + '.tif')
        iio.mimwrite(output_fn + '.tif', imvol)

        if z0s is not None:  
            z0s_fn = output_fn + '_z0s.tif'
            print(z0s_fn)
            iio.imwrite(z0s_fn, z0s)
        
        if imvolsml is not None:
            sml_frames = []
            for i in range(imvolsml.shape[0]):
                sml_frames.append(imvolsml[i,:,:])
            
            sml_fn = output_fn + '_sml.tif'
            print(sml_fn)
            iio.mimwrite(sml_fn, imvolsml)

    def run(self):
        imvol = self.read_volume()
        print(imvol.shape)

        imvol_flat, z0s, imvolsml = self.flatten(imvol)

        if 'imvol_flat' in locals():
            print(imvol_flat.shape)
            self.write_output(imvol_flat, z0s=z0s, imvolsml=imvolsml)
        else:
            self.write_output(imvol)

if __name__ == '__main__':
    mod = FlattenVolumeModule(input_dict=example_input)
    mod.run()