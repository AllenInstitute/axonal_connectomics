from acpreprocessing import downsampling
import json
import random
from multiprocessing import Pool



#configs/config_downsample_tifs_487748_36_NeuN_NFH488_25X_1XPBS.json

with open("configs/config_downsample_gifs_487748_36_NeuN_NFH488_25X_1XPBS.json", 'r') as j:
     cfg = json.loads(j.read())

random.shuffle(cfg)

#with Pool(5) as p:
#	p.map(downsampling.downsample_gif, cfg)


for c in cfg:
	downsampling.downsample_gif(c)
