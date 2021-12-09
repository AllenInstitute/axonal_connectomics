
import os
from acpreprocessing.utils import io
from argschema import ArgSchemaParser, ArgSchema
from argschema.fields import NumpyArray, Boolean,Float, Int, Str
from natsort import natsorted, ns

example_input = {
    "position": 2,
    "rootDir": "/ACdata/processed/demoModules/raw/",
    "outputDir": "/ACdata/processed/demoModules/output/",
    'dsName':'ex1'
}

def sort_files(filedir):
    filelist = os.listdir(filedir)
    return natsorted(filelist, alg=ns.IGNORECASE)

def strip_file(posdir,outputdir):
    index = 0
    for inputfile in sort_files(posdir):
        if inputfile[0]=='.':
            continue
        print(inputfile)
        I = io.get_tiff_image(posdir+inputfile)
        for j in range(I.shape[0]):
            img = I[j,:,:]
            fname = outputdir + "/{0:05d}.tif".format(index)
            print(fname)
            io.save_tiff_image(img,fname)
            index +=1


class Convert2DTiffSchema(ArgSchema):
    position = Int(required=True, description='acquisition strip position number')
    rootDir = Str(required=True, description='raw tiff root directory')
    outputDir = Str(required=True, description='output directory')
    dsName = Str(default='ex1', description='dataset name')

class Convert2DTiff():
    def __init__(self, input_json=example_input):
        self.input_data = input_json.copy()

    def run(self):
        mod = ArgSchemaParser(input_data=self.input_data,schema_type=Convert2DTiffSchema)

        posdir = "%s%s_Pos%d/"%(mod.args['rootDir'],mod.args['dsName'],mod.args['position'])
        os.makedirs(mod.args['outputDir']+"2Dtiff/Pos%d"%(mod.args['position']), exist_ok=True)
        tiffDir = '%s/2Dtiff/Pos%d'%(mod.args['outputDir'],mod.args['position'])
        os.makedirs(tiffDir, exist_ok=True)
        #if 2Dtiffs don't exist:
        if os.listdir(tiffDir) == []:
            strip_file(posdir, tiffDir)
        print("Finished 2D tiffs for Pos%d"%(mod.args['position']))


if __name__ == '__main__':
    mod = Convert2DTiff()
    mod.run()