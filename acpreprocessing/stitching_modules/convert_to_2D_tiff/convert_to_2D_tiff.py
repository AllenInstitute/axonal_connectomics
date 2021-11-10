
import os
from acpreprocessing.utils import io

from argschema import ArgSchemaParser, ArgSchema
from argschema.fields import NumpyArray, Boolean,Float, Int, Str
from natsort import natsorted, ns

example_input = {
    "position": 0,
    "rootDir": "/m2_data/iSPIM2/test/",
    "tiffDir": "/ACdata/processed/testModules/testconvert2D/",
    'dsName':'ex1'
}

def stripfile(filelist, outputdir, filedir):
    index = 0
    for inputfile in filelist:
        print(inputfile)
        I = io.get_tiff_image(filedir+inputfile)
        for j in range(I.shape[0]):
            img = I[j,:,:]
            fname = outputdir + "/{0:05d}.tif".format(index)
            print(fname)
            io.save_tiff_image(img,fname)
            index +=1

class Convert2DTiffSchema(ArgSchema):
    position = Int(required=True, description='acquisition strip position number')
    rootDir = Str(required=True, description='raw tiff root directory')
    tiffDir = Str(required=True, description='2D tiff directory')
    dsName = Str(required=True, description='dataset name')    

class Convert2DTiff():
    def run(self):
        mod = ArgSchemaParser(input_data=example_input,schema_type=Convert2DTiffSchema)

        filedir = "%s%s_Pos%d/"%(mod.args['rootDir'],mod.args['dsName'],mod.args['position'])
        filelist = os.listdir(filedir)
        files = natsorted(filelist, alg=ns.IGNORECASE)
        tiffdir = '%s/Pos%d'%(mod.args['tiffDir'],mod.args['position'])
        os.makedirs(tiffdir, exist_ok=True)
        stripfile(files, tiffdir, filedir)
        print("Finished 2D tiffs for Pos%d"%(mod.args['position']))


if __name__ == '__main__':
    mod = Convert2DTiff()
    mod.run()