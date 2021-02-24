from acpreprocessing import downsample_gif
import json
import random
from multiprocessing import Pool



#configs/config_downsample_tifs_487748_36_NeuN_NFH488_25X_1XPBS.json

with open("configs/test36.json", 'r') as j:
     cfg = json.loads(j.read())

random.shuffle(cfg)

with Pool(5) as p:
	p.map(downsample_gif.downsample_tif, cfg)


#for c in cfg:
#	downsample_tiffs_and_extract_metadata.downsample_tiffs_and_extract_metadata(c)
