import os

def add_multiscale_attributes(output_root, pixelResolution,Position):
    curdir = os.getcwd()
    os.chdir(output_root)
    attributes = open("attributes.json", "w")
    attributes.write('{"pixelResolution" : {"unit":"um","dimensions":[%f,%f,%f]},"scales":[[1,1,1],[2,2,2],[4,4,4],[8,8,8],[16,16,16],[32,32,32],[64,64,64],[128,128,128]]}'%(pixelResolution[0], pixelResolution[1], pixelResolution[2]))
    attributes.close()
    os.system('cp attributes.json %s/Pos%d/multirespos%d'%(output_root, Position, Position))
    os.chdir(output_root + 'Pos%d/multirespos%d'%(Position, Position))
    os.system('ln -s ../pos%d s0'%(Position))
    os.chdir(output_root + 'Pos%d/pos%d'%(Position, Position))
    os.system('ln -s ../pos%d pos%d'%(Position, Position))
    os.chdir(curdir)

